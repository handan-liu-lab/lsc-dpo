"""The research preference data path, retaining Handbook and HH/Qwen behavior.

`apply_preference_chat_template` is adapted from the Alignment Handbook, Copyright 2023 The HuggingFace
Team, licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0), and
modified by the LSC-DPO Authors, 2026.
"""

import random

from data_utils import get_datasets, get_tokenizer
from data_utils.data import (
    is_openai_format,
    maybe_insert_system_message,
    resolve_chat_template_kwargs,
)
from data_utils.dtypes import normalize_torch_dtype

PREFERENCE_COLUMNS = ["messages", "chosen", "rejected", "prompt", "completion", "label"]


def apply_preference_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True,
    disable_thinking: bool = False,
    model_name_or_path: str | None = None,
):
    if not all(key in example.keys() for key in ("chosen", "rejected")):
        raise ValueError(
            "Could not format example as dialogue for preference training; expected either "
            "`[chosen, rejected]` or `[prompt, chosen, rejected]`."
        )
    if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
        raise ValueError("Preference training expects OpenAI-style message dictionaries.")

    if "prompt" in example and is_openai_format(example["prompt"]):
        prompt_messages = [dict(message) for message in example["prompt"]]
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
    else:
        prompt_messages = [dict(message) for message in example["chosen"][:-1]]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

    if auto_insert_empty_system_msg:
        maybe_insert_system_message(prompt_messages, tokenizer)

    chat_template_kwargs = resolve_chat_template_kwargs(
        tokenizer,
        disable_thinking=disable_thinking,
        model_name_or_path=model_name_or_path,
    )

    text_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, **chat_template_kwargs)
    text_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False, **chat_template_kwargs)
    if tokenizer.bos_token and text_chosen.startswith(tokenizer.bos_token):
        text_chosen = text_chosen[len(tokenizer.bos_token) :]
    text_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False, **chat_template_kwargs)
    if tokenizer.bos_token and text_rejected.startswith(tokenizer.bos_token):
        text_rejected = text_rejected[len(tokenizer.bos_token) :]
    return {
        "text_prompt": text_prompt,
        "text_chosen": text_chosen,
        "text_rejected": text_rejected,
    }


def _is_main_process(training_args) -> bool:
    return getattr(training_args, "process_index", 0) == 0


def _log_processed_train_sample(raw_datasets, training_args, run_logger) -> None:
    if "train" not in raw_datasets or len(raw_datasets["train"]) == 0 or not _is_main_process(training_args):
        return

    train_dataset = raw_datasets["train"]
    index = random.Random(training_args.seed).sample(range(len(train_dataset)), k=1)[0]
    sample = train_dataset[index]
    run_logger.info(
        "Processed train sample %(sample_index)s:\n\nPrompt:\n%(prompt)s\n\nChosen:\n%(chosen)s\n\nRejected:\n%(rejected)s",
        {
            "sample_index": index,
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        },
    )


def prepare_preference_datasets(model_args, data_args, training_args, run_logger):
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=PREFERENCE_COLUMNS,
        is_main_process=_is_main_process(training_args),
        run_logger=run_logger,
    )
    run_logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    raw_datasets = raw_datasets.map(
        apply_preference_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "disable_thinking": data_args.disable_thinking,
            "model_name_or_path": model_args.model_name_or_path,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in raw_datasets.keys():
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    _log_processed_train_sample(raw_datasets, training_args, run_logger)

    training_args.model_init_kwargs = build_model_init_kwargs(model_args, training_args)
    return raw_datasets, tokenizer


def build_model_init_kwargs(model_args, training_args):
    return {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "torch_dtype": normalize_torch_dtype(model_args.torch_dtype),
        "use_cache": False if training_args.gradient_checkpointing else True,
        "attn_implementation": model_args.attn_implementation,
    }
