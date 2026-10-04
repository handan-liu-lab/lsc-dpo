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

import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import DataArguments

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
HH_TURN_PATTERN = re.compile(r"(?:^|\n\n)(Human|Assistant): ")


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        chat_template = getattr(tokenizer, "default_chat_template", None)
    if chat_template is None:
        return

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def _is_qwen_model_reference(model_name_or_path: Optional[str]) -> bool:
    return isinstance(model_name_or_path, str) and "qwen" in model_name_or_path.lower()


def resolve_chat_template_kwargs(
    tokenizer,
    *,
    disable_thinking: bool = False,
    model_name_or_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not disable_thinking:
        return {}

    tokenizer_name_or_path = getattr(tokenizer, "name_or_path", None)
    if _is_qwen_model_reference(model_name_or_path) or _is_qwen_model_reference(tokenizer_name_or_path):
        return {"enable_thinking": False}

    return {}


def parse_hh_transcript(transcript: str) -> List[Dict[str, str]]:
    if not isinstance(transcript, str):
        raise ValueError(f"HH transcript must be a string, got {type(transcript)}")

    matches = list(HH_TURN_PATTERN.finditer(transcript))
    if not matches:
        raise ValueError("HH transcript does not contain any 'Human:' or 'Assistant:' turns.")

    messages = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(transcript)
        content = transcript[start:end].strip()
        if not content:
            continue

        role = "user" if match.group(1) == "Human" else "assistant"
        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("HH transcript does not contain any non-empty turns.")

    return messages


def is_raw_hh_preference_example(example: Dict[str, Any]) -> bool:
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return False

    turn_markers = ("Human:", "Assistant:")
    return all(marker in chosen for marker in turn_markers) and all(marker in rejected for marker in turn_markers)


def _split_hh_prompt_and_responses(
    chosen_messages: List[Dict[str, str]],
    rejected_messages: List[Dict[str, str]],
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    prefix_len = 0
    for chosen_message, rejected_message in zip(chosen_messages, rejected_messages):
        if chosen_message != rejected_message:
            break
        prefix_len += 1

    if prefix_len == 0:
        raise ValueError("HH chosen/rejected transcripts must share a non-empty common prefix.")
    if prefix_len >= len(chosen_messages) or prefix_len >= len(rejected_messages):
        raise ValueError("HH chosen/rejected transcripts must each contain a divergent assistant response.")

    prompt_messages = chosen_messages[:prefix_len]
    chosen_suffix = chosen_messages[prefix_len:]
    rejected_suffix = rejected_messages[prefix_len:]

    if prompt_messages[-1]["role"] != "user":
        raise ValueError("HH prompt prefix must end with a user turn before the compared assistant responses.")
    if len(chosen_suffix) != 1 or len(rejected_suffix) != 1:
        raise ValueError("HH preprocessing expects exactly one final assistant response in chosen/rejected suffixes.")
    if chosen_suffix[0]["role"] != "assistant" or rejected_suffix[0]["role"] != "assistant":
        raise ValueError("HH chosen/rejected suffixes must each contain one assistant message.")

    return prompt_messages, chosen_suffix, rejected_suffix


def get_hh_preference_validation_error(example: Dict[str, Any]) -> Optional[str]:
    if not is_raw_hh_preference_example(example):
        return None

    try:
        chosen_messages = parse_hh_transcript(example["chosen"])
        rejected_messages = parse_hh_transcript(example["rejected"])
        _split_hh_prompt_and_responses(chosen_messages, rejected_messages)
    except ValueError as error:
        return str(error)

    return None


def maybe_convert_hh_to_openai_format(example: Dict[str, Any]) -> Dict[str, Any]:
    if not is_raw_hh_preference_example(example):
        return example

    chosen_messages = parse_hh_transcript(example["chosen"])
    rejected_messages = parse_hh_transcript(example["rejected"])
    prompt_messages, chosen_suffix, rejected_suffix = _split_hh_prompt_and_responses(
        chosen_messages,
        rejected_messages,
    )

    converted = dict(example)
    converted["prompt"] = prompt_messages
    converted["chosen"] = chosen_suffix
    converted["rejected"] = rejected_suffix
    return converted


def normalize_raw_hh_preference_dataset(
    dataset: Dataset,
    *,
    split_name: str,
    is_main_process: bool = True,
    run_logger=None,
) -> Dataset:
    active_logger = run_logger or logger
    invalid_indices = []
    invalid_reasons = Counter()

    for index, example in enumerate(dataset):
        error = get_hh_preference_validation_error(example)
        if error is None:
            continue
        invalid_indices.append(index)
        invalid_reasons[error] += 1

    if invalid_indices:
        invalid_index_set = set(invalid_indices)
        keep_indices = [index for index in range(len(dataset)) if index not in invalid_index_set]
        dataset = dataset.select(keep_indices)

        if is_main_process:
            reason_summary = ", ".join(f"{count} x {reason}" for reason, count in invalid_reasons.most_common())
            active_logger.warning(
                "Dropped %s non-canonical HH preference examples from split `%s` before normalization (%s).",
                len(invalid_indices),
                split_name,
                reason_summary,
            )

    if len(dataset) == 0:
        return Dataset.from_dict({"prompt": [], "chosen": [], "rejected": []})

    return dataset.map(
        lambda example: maybe_convert_hh_to_openai_format(dict(example)),
        remove_columns=list(dataset.column_names),
        desc=f"Normalizing raw HH preferences ({split_name})",
        load_from_cache_file=False,
    )


def maybe_normalize_preference_dataset(
    dataset: Dataset,
    *,
    dataset_name: str,
    split_name: str,
    is_main_process: bool = True,
    run_logger=None,
) -> Dataset:
    if dataset_name != "Anthropic/hh-rlhf":
        return dataset

    return normalize_raw_hh_preference_dataset(
        dataset,
        split_name=split_name,
        is_main_process=is_main_process,
        run_logger=run_logger,
    )


def _columns_to_keep_for_dataset(dataset: Dataset, columns_to_keep: List[str]) -> List[str]:
    requested_columns = set(columns_to_keep)
    canonical_preference_columns = {"prompt", "chosen", "rejected"}

    if canonical_preference_columns.intersection(requested_columns) and canonical_preference_columns.issubset(
        set(dataset.column_names)
    ):
        requested_columns.update(canonical_preference_columns)

    return [column for column in dataset.column_names if column in requested_columns]


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


def get_datasets(
    data_config: DataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    is_main_process: bool = True,
    run_logger=None,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
        is_main_process=is_main_process,
        run_logger=run_logger,
        revisions=getattr(data_config, "dataset_revisions", None),
    )
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
    is_main_process: bool = True,
    run_logger=None,
    revisions: Optional[Dict[str, str]] = None,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            local_saved_split = Path(ds).expanduser() / split
            if (local_saved_split / "dataset_info.json").is_file() and (local_saved_split / "state.json").is_file():
                dataset = load_from_disk(str(local_saved_split))
            else:
                try:
                    # Keep the existing API surface and only reinterpret `dataset_configs`
                    # as `data_dir` for Anthropic HH raw subsets such as `helpful-base`.
                    if ds == "Anthropic/hh-rlhf" and ds_config is not None:
                        dataset = load_dataset(
                            ds,
                            split=split,
                            data_dir=ds_config,
                            **({"revision": revisions[ds]} if revisions and ds in revisions else {}),
                        )
                    else:
                        dataset = load_dataset(
                            ds,
                            ds_config,
                            split=split,
                            **({"revision": revisions[ds]} if revisions and ds in revisions else {}),
                        )
                except DatasetGenerationError:
                    # Preserve the historical fallback for other local dataset layouts.
                    dataset = load_from_disk(os.path.join(ds, split))

            dataset = maybe_normalize_preference_dataset(
                dataset,
                dataset_name=ds,
                split_name=split,
                is_main_process=is_main_process,
                run_logger=run_logger,
            )

            # Remove redundant columns to avoid schema conflicts on load.
            columns_to_keep_for_dataset = _columns_to_keep_for_dataset(dataset, columns_to_keep)
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in columns_to_keep_for_dataset]
            )
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
