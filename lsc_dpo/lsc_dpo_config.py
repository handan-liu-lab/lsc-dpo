from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional

from transformers import TrainingArguments


@dataclass
class LSCDPOConfig(TrainingArguments):
    r"""
    LSCDPOConfig collects all training arguments related to the [`LSCDPOTrainer`] class.
    The same config trains LSC-DPO and fixed-beta DPO; the two methods share data,
    scoring, optimizer and checkpoint behavior.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        method (`str`, defaults to `lsc_dpo`):
            The training method, either `lsc_dpo` or `dpo`. `dpo` trains with a constant beta and ignores
            `beta_0`, `z_star`, `tau_start` and `eta`.
        beta_0 (`float`, defaults to 0.01):
            The initial beta for LSC-DPO. The controller updates beta from this value during training.
        z_star (`float`, defaults to 0.4):
            The target scaled margin z*. The controller keeps beta * (mean margin) near z*, which keeps the
            learning signal sigmoid(-beta * m) near sigmoid(-z*), about 0.40 for the default.
        tau_start (`float`, defaults to 0.45):
            The start threshold. Beta is updated only while the learning signal is below `tau_start`.
            At the start of training every margin is 0 and the signal is sigmoid(0) = 0.5.
        eta (`float`, defaults to 0.1):
            The step size of the multiplicative beta update. The update is locally stable when
            0 < eta * z_star < 2.
        beta (`float`, *optional*):
            The constant beta for `method=dpo`. Defaults to 0.1. For LSC-DPO, use `beta_0` instead.
        max_length (`int`, defaults to 512):
            The maximum length of the prompt plus response, in tokens.
        max_prompt_length (`int`, defaults to 128):
            The maximum length of the prompt. Must be smaller than `max_length`.
        truncation_mode (`str`, defaults to `keep_end`):
            How to truncate a long prompt, either `keep_end` or `keep_start`.
        label_pad_token_id (`int`, defaults to `-100`):
            The label for prompt and padding tokens. Tokens with this label are not scored.
        padding_value (`int`, *optional*):
            The padding value if it is different to the tokenizer's pad_token_id.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in the policy and reference models.
        non_finite_logits_handling (`str`, defaults to `sanitize`):
            What to do with NaN or infinite logits: `sanitize` replaces them and warns, `error` stops training.
        dataset_num_proc (`int`, *optional*):
            The number of workers to use to tokenize the data.
        tokenization_batch_size (`int`, defaults to 128):
            The number of rows tokenized per batch.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string.
        remove_unused_columns (`bool`, defaults to `False`):
            Must stay `False`: the trainer needs the tokenized chosen and rejected columns.
        save_hf_model_artifacts (`bool`, defaults to `True`):
            Whether to write a final Hugging Face model export after training.
        logging_first_step (`bool`, defaults to `True`):
            Whether to log the first global step.
    """

    method: str = "lsc_dpo"
    beta_0: float = 0.01
    z_star: float = 0.4
    tau_start: float = 0.45
    eta: float = 0.1
    beta: Optional[float] = None

    max_length: int = 512
    max_prompt_length: int = 128
    truncation_mode: str = "keep_end"
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    disable_dropout: bool = True
    non_finite_logits_handling: str = "sanitize"

    dataset_num_proc: Optional[int] = None
    tokenization_batch_size: int = 128

    model_init_kwargs: Optional[Dict[str, Any]] = field(default=None, repr=False)
    remove_unused_columns: bool = False
    save_hf_model_artifacts: bool = True
    logging_first_step: bool = True

    @staticmethod
    def normalize_input_dict(values):
        # H4ArgumentParser reconstructs a parsed dataclass for YAML overrides.
        # Drop the derived beta so --beta_0 changes the canonical input cleanly.
        values = dict(values)
        if values.get("method", "lsc_dpo") == "lsc_dpo" and "beta_0" in values:
            if values.get("beta") == values["beta_0"]:
                values.pop("beta", None)
        return values

    def __post_init__(self):
        # Handbook YAML overrides leave Optional[float] values as strings.
        if self.beta is not None:
            self.beta = float(self.beta)
        if self.method not in {"lsc_dpo", "dpo"}:
            raise ValueError("method must be lsc_dpo or dpo.")
        if self.method == "lsc_dpo":
            if self.beta is not None and self.beta != self.beta_0:
                raise ValueError("Use beta_0 for LSC-DPO; conflicting beta is not supported.")
            if not all(math.isfinite(x) for x in (self.beta_0, self.z_star, self.tau_start, self.eta)):
                raise ValueError("LSC coefficients must be finite.")
            if self.eta < 0 or not 0 <= self.tau_start <= 1:
                raise ValueError("Require eta >= 0 and 0 <= tau_start <= 1.")
            # The trainer reads self.beta, so LSC-DPO starts it at beta_0.
            self.beta = self.beta_0
        elif self.beta is None:
            self.beta = 0.1
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if self.max_length <= 0 or not 0 < self.max_prompt_length < self.max_length:
            raise ValueError("Require 0 < max_prompt_length < max_length.")
        if self.truncation_mode not in {"keep_start", "keep_end"}:
            raise ValueError("truncation_mode must be keep_start or keep_end.")
        if self.tokenization_batch_size <= 0:
            raise ValueError("tokenization_batch_size must be positive.")
        if self.non_finite_logits_handling not in {"sanitize", "error"}:
            raise ValueError("non_finite_logits_handling must be sanitize or error.")
        if self.push_to_hub or self.deepspeed or self.remove_unused_columns:
            raise ValueError("Use local saves, FSDP/single-device training and remove_unused_columns=false.")
        super().__post_init__()

    def to_dict(self):
        values = super().to_dict()
        # Save only the coefficients used by the selected method.
        for name in ["beta"] if self.method == "lsc_dpo" else ["beta_0", "z_star", "tau_start", "eta"]:
            values.pop(name, None)
        return values
