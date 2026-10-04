# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Modified by the LSC-DPO Authors, 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .configs import DataArguments, ModelArguments
from .data import DEFAULT_CHAT_TEMPLATE


# Enable parallel Hub downloads by default when hf_transfer is installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def get_active_chat_template(tokenizer: PreTrainedTokenizer) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(tokenizer, "default_chat_template", None)
    return chat_template or ""


def has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    return bool(get_active_chat_template(tokenizer))


def get_tokenizer(
    model_args: ModelArguments, data_args: DataArguments, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.model_name_or_path
            if model_args.tokenizer_name_or_path is None
            else model_args.tokenizer_name_or_path
        ),
        revision=model_args.tokenizer_revision or model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif auto_set_chat_template and not has_chat_template(tokenizer):
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_checkpoint(training_args: TrainingArguments) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def _find_state_tensor(state_dict: Mapping[str, torch.Tensor], suffix: str) -> torch.Tensor | None:
    tensor = state_dict.get(suffix)
    if tensor is not None:
        return tensor
    for key, value in state_dict.items():
        if key.endswith(suffix):
            return value
    return None


def validate_causal_lm_export(
    model: Any,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return

    expected_vocab_size = getattr(config, "vocab_size", None)
    hidden_size = getattr(config, "hidden_size", None)
    if expected_vocab_size is None or hidden_size is None:
        return

    embed_tokens = _find_state_tensor(state_dict, "model.embed_tokens.weight")
    if embed_tokens is not None:
        expected_shape = (expected_vocab_size, hidden_size)
        actual_shape = tuple(embed_tokens.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                "Refusing to save an invalid checkpoint: "
                f"expected model.embed_tokens.weight shape {expected_shape}, got {actual_shape}. "
                "This usually means the save path exported a sharded/flattened FSDP tensor instead of a full HF checkpoint."
            )

    lm_head = _find_state_tensor(state_dict, "lm_head.weight")
    if lm_head is not None and lm_head.numel() > 0:
        expected_shape = (expected_vocab_size, hidden_size)
        actual_shape = tuple(lm_head.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                "Refusing to save an invalid checkpoint: "
                f"expected lm_head.weight shape {expected_shape}, got {actual_shape}. "
                "This checkpoint would not load cleanly for inference."
            )
