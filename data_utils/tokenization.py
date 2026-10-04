"""Tokenize preference pairs for DPO, keeping the prompt/response boundary exact.

Adapted from TRL's DPOTrainer tokenization, Copyright 2023 The HuggingFace Team, licensed under the
Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Modified by the LSC-DPO Authors, 2026.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

import numpy as np


@dataclass
class PreferenceTokenizationProcessor:
    r"""
    Tokenize (prompt, chosen, rejected) rows into input ids and labels.

    Prompt tokens are labeled with `label_pad_token_id`, so only response tokens are scored.
    """

    tokenizer: Any
    max_length: int
    max_prompt_length: int
    truncation_mode: str
    label_pad_token_id: int
    long_sequence_warning_key: str

    def _tokenize_without_max_length_warning(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        previous = self.tokenizer.deprecation_warnings.get(self.long_sequence_warning_key, False)
        self.tokenizer.deprecation_warnings[self.long_sequence_warning_key] = True
        try:
            return self.tokenizer(texts, **kwargs)
        finally:
            self.tokenizer.deprecation_warnings[self.long_sequence_warning_key] = previous

    def tokenize_batch(self, features: Dict[str, Sequence[Any]]) -> Dict[str, List[Any]]:
        """Tokenize a batch of rows; used with `datasets.Dataset.map(batched=True)`."""
        prompts = list(features["prompt"])
        chosen = list(features["chosen"])
        rejected = list(features["rejected"])
        for name, values in (("prompt", prompts), ("chosen", chosen), ("rejected", rejected)):
            if not all(isinstance(value, str) for value in values):
                bad_type = next(type(value) for value in values if not isinstance(value, str))
                raise ValueError(f"{name} should be an str but got {bad_type}")

        prompt_batch = self._tokenize_without_max_length_warning(prompts, add_special_tokens=False)
        chosen_full_batch = self._tokenize_without_max_length_warning(
            [prompt + answer for prompt, answer in zip(prompts, chosen)],
            add_special_tokens=False,
        )
        rejected_full_batch = self._tokenize_without_max_length_warning(
            [prompt + answer for prompt, answer in zip(prompts, rejected)],
            add_special_tokens=False,
        )

        output: Dict[str, List[Any]] = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
        }

        for idx in range(len(prompts)):
            prompt_tokens = {
                "prompt_input_ids": list(prompt_batch["input_ids"][idx]),
                "prompt_attention_mask": list(prompt_batch["attention_mask"][idx]),
            }
            chosen_tokens = self._build_answer_tokens_from_full(
                prompt_input_ids=prompt_tokens["prompt_input_ids"],
                prompt_attention_mask=prompt_tokens["prompt_attention_mask"],
                full_input_ids=list(chosen_full_batch["input_ids"][idx]),
                full_attention_mask=list(chosen_full_batch["attention_mask"][idx]),
            )
            rejected_tokens = self._build_answer_tokens_from_full(
                prompt_input_ids=prompt_tokens["prompt_input_ids"],
                prompt_attention_mask=prompt_tokens["prompt_attention_mask"],
                full_input_ids=list(rejected_full_batch["input_ids"][idx]),
                full_attention_mask=list(rejected_full_batch["attention_mask"][idx]),
            )

            example = self._finalize_decoder_only_example(prompt_tokens, chosen_tokens, rejected_tokens)
            for key, value in example.items():
                output[key].append(value)

        return output

    def _build_answer_tokens_from_full(
        self,
        prompt_input_ids: List[int],
        prompt_attention_mask: List[int],
        full_input_ids: List[int],
        full_attention_mask: List[int],
    ) -> Dict[str, List[int]]:
        """
        Llama tokenizer does not satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        answer_input_ids = full_input_ids[len(prompt_input_ids) :]
        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids_array = np.array(full_input_ids)

        if len(full_input_ids_array) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # Some tokenizers merge the last prompt token with the first answer token when
        # tokenizing prompt+answer. If the prompt tokens no longer match, the response
        # starts one token earlier.
        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_input_ids[:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        corrected_prompt_input_ids = full_input_ids[:response_token_ids_start_idx]
        corrected_prompt_attention_mask = full_attention_mask[:response_token_ids_start_idx]

        if len(corrected_prompt_input_ids) != len(corrected_prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        corrected_answer_input_ids = full_input_ids[response_token_ids_start_idx:]
        corrected_answer_attention_mask = full_attention_mask[response_token_ids_start_idx:]

        return {
            "prompt_input_ids": corrected_prompt_input_ids,
            "prompt_attention_mask": corrected_prompt_attention_mask,
            "input_ids": corrected_answer_input_ids,
            "attention_mask": corrected_answer_attention_mask,
        }

    def _finalize_decoder_only_example(
        self,
        prompt_tokens: Dict[str, List[int]],
        chosen_tokens: Dict[str, List[int]],
        rejected_tokens: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """Add BOS/EOS, truncate to `max_length`, and build labels for one preference pair."""
        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        prompt_tokens = {key: value[:prompt_len_input_ids] for key, value in prompt_tokens.items()}
        chosen_tokens = {key: list(value) for key, value in chosen_tokens.items()}
        rejected_tokens = {key: list(value) for key, value in rejected_tokens.items()}

        # Make sure prompts differ in at most one token and their lengths by at most one
        num_diff_tokens = sum(
            a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt. Avoid adding if it's already there or the tokenizer has none
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is not None:
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for key in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[key] = answer_tokens[key][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for key in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[key] = answer_tokens[key][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for key in ["input_ids", "attention_mask"]:
                    answer_tokens[key] = answer_tokens[key][: self.max_length - self.max_prompt_length]

        # Create labels: prompt positions get label_pad_token_id so only the response is scored
        chosen_sequence_tokens = {
            key: chosen_tokens[f"prompt_{key}"] + chosen_tokens[key] for key in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            key: rejected_tokens[f"prompt_{key}"] + rejected_tokens[key] for key in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(
            chosen_tokens["prompt_input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        example: Dict[str, List[int]] = {}
        for prefix, tokens in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for key, value in tokens.items():
                if key == "token_type_ids":
                    continue
                example[f"{prefix}{key}"] = value
        return example
