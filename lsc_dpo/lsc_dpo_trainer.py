# The chosen/rejected scoring (concatenated_inputs, concatenated_forward, get_batch_logps) and the
# metric logging (store_metrics, log) are adapted from TRL's DPOTrainer, Copyright 2023 The HuggingFace
# Team, licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0), and modified
# by the LSC-DPO Authors, 2026. The rest of this file is MIT-licensed. See README.md#license.
from __future__ import annotations

import json
import math
import pickle
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType
from transformers import AutoModelForCausalLM, Trainer
from transformers.trainer import ParallelMode, set_rng_state_for_device
from transformers.trainer_utils import get_last_checkpoint
from trl.trainer.utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length

from data_utils.dtypes import normalize_torch_dtype
from data_utils.model_utils import validate_causal_lm_export
from data_utils.tokenization import PreferenceTokenizationProcessor
from lsc_dpo.lsc_dpo_config import LSCDPOConfig


def update_beta(beta: float, *, q_t: float, margin_mean: float, z_star: float, tau_start: float, eta: float) -> float:
    """Update beta with one LSC feedback step.

    The controller moves the scaled margin beta * margin_mean toward z_star, which keeps the
    learning signal sigmoid(-beta * m) near sigmoid(-z_star). If the scaled margin is above
    z_star, the signal is too small and beta shrinks; if it is below, beta grows. The
    multiplicative form keeps beta positive. The controller acts only while q_t < tau_start,
    and the check is repeated on every call.

    Args:
        beta: Current coefficient, the one used to compute q_t.
        q_t: Learning signal, the mean of sigmoid(-beta * m) over the batch.
        margin_mean: Mean preference margin m over the batch.
        z_star: Target scaled margin.
        tau_start: Update only while q_t < tau_start.
        eta: Step size of the multiplicative update.

    Returns:
        The beta for this microbatch's loss: beta * exp(-eta * (beta * margin_mean - z_star)),
        or beta unchanged when q_t >= tau_start.
    """
    if q_t < tau_start:
        delta = -(beta * margin_mean - z_star)
        beta = beta * math.exp(delta * eta)
    if not math.isfinite(beta) or beta <= 0:
        raise FloatingPointError("LSC produced a non-finite or non-positive beta.")
    return beta


class LSCDPOTrainer(Trainer):
    """Full-parameter DPO trainer; `method="lsc_dpo"` puts beta under feedback control.

    `self.beta` is the coefficient in use. LSC-DPO starts it at `beta_0` and calls `update_beta`
    on every training microbatch; fixed-beta DPO leaves it at `beta`.

    Args:
        model: Policy to train, as a model or a Hugging Face ID or local path.
        ref_model: Frozen reference, usually the same SFT checkpoint as `model` but a separate instance.
        args: An `LSCDPOConfig` naming the method and its coefficients.
        train_dataset: Preference pairs with `prompt`, `chosen` and `rejected` columns.
        eval_dataset: Held-out pairs in the same format.
        tokenizer: Shared by the policy and the reference.
    """

    _LONG_SEQUENCE_WARNING_KEY = "sequence-length-is-longer-than-the-specified-maximum"

    # Loading the policy and reference, and preparing them for distributed training

    def __init__(
        self, model, ref_model, args: LSCDPOConfig, train_dataset=None, eval_dataset=None, tokenizer=None, **kwargs
    ):
        if not isinstance(args, LSCDPOConfig):
            raise TypeError("Use LSCDPOConfig for both training methods.")
        if tokenizer is None or ref_model is None:
            raise ValueError("A tokenizer and explicit reference model are required.")
        model_kwargs = dict(args.model_init_kwargs or {})
        if model_kwargs and not isinstance(model, str):
            raise ValueError("model_init_kwargs requires a model path, not an instantiated model.")
        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = normalize_torch_dtype(model_kwargs["torch_dtype"])
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self.ref_model = (
            AutoModelForCausalLM.from_pretrained(ref_model, **model_kwargs)
            if isinstance(ref_model, str)
            else ref_model
        )
        if model.config.is_encoder_decoder or self.ref_model.config.is_encoder_decoder:
            raise ValueError("Only decoder-only models are supported.")
        if model is self.ref_model:
            raise ValueError("Policy and reference must be separate model instances.")
        if args.disable_dropout:
            disable_dropout_in_model(model)
            disable_dropout_in_model(self.ref_model)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()

        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = tokenizer.pad_token_id if args.padding_value is None else args.padding_value
        self.non_finite_logits_handling = args.non_finite_logits_handling
        self.tokenizer = tokenizer
        self.tokenization_processor = PreferenceTokenizationProcessor(
            tokenizer,
            args.max_length,
            args.max_prompt_length,
            args.truncation_mode,
            args.label_pad_token_id,
            self._LONG_SEQUENCE_WARNING_KEY,
        )
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.beta = float(args.beta)
        self._controller_clock_global_step = None
        self._controller_microbatch_index = 0
        self._policy_model_prepared_for_evaluation = False

        train_dataset = self._tokenize_dataset(train_dataset, "train", args)
        eval_dataset = self._tokenize_dataset(eval_dataset, "test", args)
        collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id, label_pad_token_id=args.label_pad_token_id, is_encoder_decoder=False
        )
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            **kwargs,
        )

        # FSDP's CPU-efficient load does not synchronize the unwrapped reference model.
        self.ref_model = self.accelerator.prepare_model(self.ref_model, device_placement=True, evaluation_mode=True)
        self._sync_unwrapped_model_from_rank0(self.ref_model, model_name="reference")
        if self.is_fsdp_enabled and args.do_eval and not args.do_train:
            self._prepare_policy_model_for_evaluation()

    def _tokenize_dataset(self, dataset, split_name: str, args: LSCDPOConfig):
        if dataset is None:
            return None
        return dataset.map(
            self.tokenization_processor.tokenize_batch,
            batched=True,
            batch_size=args.tokenization_batch_size,
            num_proc=args.dataset_num_proc,
            desc=f"Tokenizing {split_name}",
        )

    def _prepare_policy_model_for_evaluation(self) -> nn.Module:
        if self._policy_model_prepared_for_evaluation:
            return self.model

        model = self._wrap_model(self.model, training=False)
        if self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8":
            model = self.accelerator.prepare(model)
        else:
            model = self.accelerator.prepare_model(model, evaluation_mode=True)

        self.model = model
        self.model_wrapped = model
        self._policy_model_prepared_for_evaluation = True
        return model

    def _is_distributed_run(self) -> bool:
        return dist.is_available() and dist.is_initialized() and self.accelerator.num_processes > 1

    def _sync_unwrapped_model_from_rank0(self, model: nn.Module, model_name: str) -> None:
        if not self._is_distributed_run():
            return

        # Manually loaded auxiliary models are not FSDP-wrapped, so they do not benefit from
        # rank-0 state syncing when Accelerate's FSDP CPU-RAM-efficient loading is enabled.
        for _, tensor in model.named_parameters():
            dist.broadcast(tensor.data, src=0)
        for _, tensor in model.named_buffers():
            dist.broadcast(tensor.data, src=0)
        dist.barrier()

    # Loss and beta controller

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """Return the loss for one microbatch and the metrics to log.

        For each pair (x, y+, y-), the preference margin is
            m = [log pi(y+|x) - log pi_ref(y+|x)] - [log pi(y-|x) - log pi_ref(y-|x)]
        and the loss is -log sigmoid(beta * m). The factor sigmoid(-beta * m) in its gradient is the
        learning signal, which fades as margins grow. During LSC-DPO training, `update_beta` adjusts
        beta before the loss is computed, to keep that signal near its target.
        """
        chosen, rejected, chosen_logits, rejected_logits, _ = self.concatenated_forward(
            model, batch, average_log_prob=False, logit_source="policy"
        )
        ref_chosen, ref_rejected = self._get_reference_logps(batch)
        ref_chosen, ref_rejected = ref_chosen.to(chosen.device), ref_rejected.to(chosen.device)
        # Form each log-ratio before subtracting, preserving reference floating-point order.
        margin = (chosen - ref_chosen) - (rejected - ref_rejected)
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}
        if self.args.method == "lsc_dpo" and train_eval == "train":
            self._advance_controller_clock()
            # The controller sees all ranks once per microbatch, using beta BEFORE updating.
            global_margin = self.accelerator.gather_for_metrics(margin.detach()).float()
            q_t = float(torch.sigmoid(-float(self.beta) * global_margin).mean().item())
            self.beta = update_beta(
                float(self.beta),
                q_t=q_t,
                margin_mean=float(global_margin.mean().item()),
                z_star=self.args.z_star,
                tau_start=self.args.tau_start,
                eta=self.args.eta,
            )
            metrics["lsc_dpo/q_t"] = torch.tensor(q_t, device=margin.device)
        # Fixed-beta DPO and evaluation both skip the controller mutation.
        loss = -F.logsigmoid(float(self.beta) * margin).mean()
        metrics[f"{prefix}dpo/beta"] = torch.tensor(self.beta, device=margin.device)
        for name, values in (
            ("logps/chosen", chosen),
            ("logps/rejected", rejected),
            ("logps/ref_chosen", ref_chosen),
            ("logps/ref_rejected", ref_rejected),
            ("logits/chosen", chosen_logits),
            ("logits/rejected", rejected_logits),
            ("dpo/margin", margin),
        ):
            metrics[prefix + name] = self._gather_mean_metric(values)
        return loss, metrics

    def _advance_controller_clock(self) -> None:
        """Advance the controller clock by one microbatch.

        Beta is updated once per microbatch, so `gradient_accumulation_steps` times per optimizer step.
        Raises if more microbatches arrive before the optimizer step, since checkpoints assume that count.
        """
        completed_step = int(self.state.global_step)
        if self._controller_clock_global_step != completed_step:
            self._controller_clock_global_step = completed_step
            self._controller_microbatch_index = 0
        if self._controller_microbatch_index >= int(self.args.gradient_accumulation_steps):
            raise RuntimeError(
                "Observed more controller microbatches than the frozen gradient "
                "accumulation count before the optimizer step advanced."
            )
        self._controller_microbatch_index += 1

    def _get_reference_logps(self, batch):
        # Sum response-token log-probabilities with exactly the policy's masking rules.
        with torch.no_grad():
            chosen, rejected, *_ = self.concatenated_forward(
                self.ref_model, batch, average_log_prob=False, logit_source="reference"
            )
        return chosen, rejected

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        self.store_metrics(metrics, train_eval="train")
        return (loss, metrics) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        self.store_metrics(metrics, train_eval="eval")
        if prediction_loss_only:
            return loss.detach(), None, None
        # Metric means are on CPU; distributed prediction gathering needs device tensors.
        logits = torch.stack([metrics["eval_logits/chosen"], metrics["eval_logits/rejected"]]).to(
            self.accelerator.device
        )
        return loss.detach(), logits, torch.zeros(logits.shape[0], device=self.accelerator.device)

    # Response log-probabilities

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Stack chosen above rejected so one forward pass scores both.

        Both halves are padded to the longer of the two sequence lengths.

        Args:
            batch: Collated pairs with `chosen_*` and `rejected_*` input ids, attention masks and
                labels, each of shape (batch_size, sequence_length).
            label_pad_token_id: Fill value for padded label positions.
            padding_value: Fill value for padded input ids.
            device: Where to place the stacked tensors.

        Returns:
            `concatenated_input_ids`, `concatenated_attention_mask` and `concatenated_labels`, each
            with 2 * batch_size rows: chosen first, then rejected.
        """
        concatenated_batch = {}
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for key in batch:
            if key.startswith("chosen") and isinstance(batch[key], torch.Tensor):
                if "labels" in key:
                    pad_value = label_pad_token_id
                elif key.endswith("_input_ids"):
                    pad_value = padding_value
                elif key.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = key.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[key],
                    max_length,
                    pad_value=pad_value,
                ).to(device=device)

        for key in batch:
            if key.startswith("rejected") and isinstance(batch[key], torch.Tensor):
                if "labels" in key:
                    pad_value = label_pad_token_id
                elif key.endswith("_input_ids"):
                    pad_value = padding_value
                elif key.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = key.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[key], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        return concatenated_batch

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        average_log_prob: bool = False,
        logit_source: str = "unknown",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        """Score chosen and rejected responses with `model` in a single forward pass.

        One pass instead of two halves the FSDP parameter gathers per microbatch.

        Returns:
            A tuple (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels).
            chosen_logps is log pi(y+|x) and rejected_logps is log pi(y-|x). Shape: (batch_size,)
        """
        model_device = self._get_model_device(model)
        if model_device is None or model_device.type == "cpu":
            model_device = self.accelerator.device
        concatenated_batch = self.concatenated_inputs(
            batch,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=model_device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        ).logits
        all_logits = self._handle_non_finite_logits(all_logits, logit_source=logit_source)

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels

    @staticmethod
    def _get_model_device(model: nn.Module) -> Optional[torch.device]:
        for tensor in model.parameters():
            return tensor.device
        for tensor in model.buffers():
            return tensor.device
        return None

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Sum (or average) next-token log-probabilities over the response tokens.

        Position t of `logits` predicts label t + 1. Positions labelled `label_pad_token_id`, the
        prompt and the padding, are left out.

        Args:
            logits: Unnormalized model outputs, shape (batch_size, sequence_length, vocab_size).
            labels: Target ids, shape (batch_size, sequence_length).
            average_log_prob: Divide by the number of response tokens instead of summing.
            label_pad_token_id: Label value marking positions to skip.

        Returns:
            Shape (batch_size,). LSC-DPO and DPO use the sum, which is log pi(y|x) for the response y.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:]
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # Masked positions need some valid index for gather; the mask zeroes them below.
        safe_labels = labels.masked_fill(~loss_mask, 0)
        per_token_logps = LSCDPOTrainer._compute_token_logps(logits, safe_labels)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1)
        return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def _compute_token_logps(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """log softmax(logits)[label] per token; logits are already finite (see `_handle_non_finite_logits`)."""
        selected_logits = torch.gather(logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        log_normalizers = torch.logsumexp(logits, dim=-1)
        per_token_logps = selected_logits - log_normalizers

        if torch.isfinite(per_token_logps).all():
            return per_token_logps

        if logits.dtype not in (torch.float16, torch.bfloat16):
            return per_token_logps

        # Half-precision logsumexp can overflow; redo it in float32, one vocabulary chunk at a time.
        selected_logits_fp32 = selected_logits.float()
        log_normalizers_fp32 = LSCDPOTrainer._chunked_logsumexp_fp32(logits)
        return selected_logits_fp32 - log_normalizers_fp32

    def _handle_non_finite_logits(self, logits: torch.FloatTensor, logit_source: str) -> torch.FloatTensor:
        if torch.isfinite(logits).all():
            return logits

        message = self._format_non_finite_logits_message(logits, logit_source)
        if self.non_finite_logits_handling == "error":
            raise ValueError(message)

        warnings.warn(
            message + " Sanitizing logits to keep reference/preference scoring numerically stable.",
            RuntimeWarning,
        )
        return LSCDPOTrainer._sanitize_logits_for_logps(logits)

    def _format_non_finite_logits_message(self, logits: torch.FloatTensor, logit_source: str) -> str:
        non_finite_count = int((~torch.isfinite(logits)).sum().item())
        nan_count = int(torch.isnan(logits).sum().item())
        posinf_count = int(torch.isposinf(logits).sum().item())
        neginf_count = int(torch.isneginf(logits).sum().item())
        return (
            f"Detected {non_finite_count} non-finite values in {logit_source} logits before token log-prob computation "
            f"(global_step={int(self.state.global_step)}, epoch={self.state.epoch}, process_index={self.accelerator.process_index}, "
            f"shape={tuple(logits.shape)}, dtype={logits.dtype}, device={logits.device}, nan={nan_count}, "
            f"posinf={posinf_count}, neginf={neginf_count})."
        )

    @staticmethod
    def _sanitize_logits_for_logps(logits: torch.FloatTensor) -> torch.FloatTensor:
        sanitized_logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return sanitized_logits.clamp_(min=-1e4, max=1e4)

    @staticmethod
    def _chunked_logsumexp_fp32(logits: torch.FloatTensor, chunk_size: int = 2048) -> torch.FloatTensor:
        log_normalizers = None
        vocab_size = logits.shape[-1]
        for start in range(0, vocab_size, chunk_size):
            chunk = logits[..., start : start + chunk_size].float()
            chunk_logsumexp = torch.logsumexp(chunk, dim=-1)
            log_normalizers = (
                chunk_logsumexp if log_normalizers is None else torch.logaddexp(log_normalizers, chunk_logsumexp)
            )
        return log_normalizers

    # Metrics

    def _gather_mean_metric(self, values: torch.Tensor) -> torch.Tensor:
        detached = values.detach()
        local_count = torch.tensor(
            float(detached.numel()),
            device=detached.device,
            dtype=torch.float64,
        )
        if detached.numel() == 0:
            local_sum = torch.zeros((), device=detached.device, dtype=torch.float64)
        else:
            local_sum = detached.mean(dtype=torch.float32).to(dtype=torch.float64) * local_count

        gathered_stats = self.accelerator.gather(torch.stack((local_sum, local_count)).reshape(1, 2))
        total_sum, total_count = gathered_stats.sum(dim=0)
        return (total_sum / total_count.clamp_min(1.0)).to(dtype=torch.float32).cpu()

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        super().log(logs, start_time)

    # Checkpoints: save and resume

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        if not getattr(self, "is_fsdp_enabled", False):
            return super().save_model(
                output_dir=output_dir,
                _internal_call=_internal_call,
            )

        fsdp_plugin = getattr(
            getattr(self.accelerator, "state", None),
            "fsdp_plugin",
            None,
        )
        uses_full_state_dict = "FULL_STATE_DICT" in str(getattr(fsdp_plugin, "state_dict_type", ""))
        if not uses_full_state_dict:
            return super().save_model(
                output_dir=output_dir,
                _internal_call=_internal_call,
            )

        if output_dir is None:
            output_dir = self.args.output_dir

        state_dict = self.accelerator.get_state_dict(self.model, unwrap=False)
        if self.args.should_save:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            validate_causal_lm_export(unwrapped_model, state_dict)
            self._save(output_dir, state_dict=state_dict)

        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def save_final_model(self, output_dir):
        """Gather wrapped FSDP weights before writing a standard HF model and tokenizer."""
        self.accelerator.wait_for_everyone()
        state = self.accelerator.get_state_dict(
            self.model, unwrap=self.accelerator.distributed_type != DistributedType.FSDP
        )
        model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            validate_causal_lm_export(model, state)
            # Match the reference export: inference should use KV caching after training.
            model.config.use_cache = True
            model.save_pretrained(
                output_dir,
                state_dict=state,
                safe_serialization=self.args.save_safetensors,
                save_function=self.accelerator.save,
            )
            self.processing_class.save_pretrained(output_dir)
        self.accelerator.wait_for_everyone()

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        if self.args.method == "lsc_dpo" and self.is_world_process_zero():
            # Beta is not a model weight, so save it beside the checkpoint for resuming.
            # Optimizer-boundary checkpoints preserve the next microbatch's coefficient.
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            state = {
                "schema_version": 1,
                "global_step": int(self.state.global_step),
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "next_controller_update_index": self.state.global_step * self.args.gradient_accumulation_steps,
                "beta_after": float(self.beta),
                "beta_policy_mode": "closed_loop_lsc",
            }
            (checkpoint / "beta_policy_state.json").write_text(json.dumps(state, indent=2) + "\n")

    def train(self, *args, resume_from_checkpoint=None, **kwargs):
        checkpoint = resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint under {self.args.output_dir}")
        if checkpoint and self.args.method == "lsc_dpo":
            # Restore beta so the controller continues where it stopped.
            saved = json.loads((Path(checkpoint) / "beta_policy_state.json").read_text())
            if saved["beta_policy_mode"] != "closed_loop_lsc":
                raise ValueError("Checkpoint method does not match LSC-DPO.")
            if saved["gradient_accumulation_steps"] != self.args.gradient_accumulation_steps:
                raise ValueError("Checkpoint gradient accumulation does not match the controller clock.")
            self.beta = float(saved["beta_after"])
        return super().train(*args, resume_from_checkpoint=checkpoint, **kwargs)

    def _load_rng_state(self, checkpoint):
        # PyTorch 2.6+ defaults to weights-only loading; local Trainer RNG files include NumPy state.
        try:
            return super()._load_rng_state(checkpoint)
        except pickle.UnpicklingError as error:
            if checkpoint is None or "Weights only load failed" not in str(error):
                raise
            name = f"rng_state_{self.args.process_index}.pth" if self.args.world_size > 1 else "rng_state.pth"
            path = Path(checkpoint) / name
            if not path.is_file():
                raise
            state = torch.load(path, weights_only=False)
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.random.set_rng_state(state["cpu"])
            if torch.cuda.is_available():
                set_rng_state_for_device(
                    "CUDA", torch.cuda, state, self.args.parallel_mode == ParallelMode.DISTRIBUTED
                )
