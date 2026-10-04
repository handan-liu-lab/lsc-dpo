#!/usr/bin/env python
"""Train LSC-DPO or fixed-beta DPO from a YAML recipe; see README.md for commands."""

import logging
import sys
from pathlib import Path

# Load the helper code from this repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import set_seed
from data_utils import DataArguments, H4ArgumentParser, ModelArguments, get_checkpoint
from data_utils.preprocessing import prepare_preference_datasets
from lsc_dpo.lsc_dpo_config import LSCDPOConfig
from lsc_dpo.lsc_dpo_trainer import LSCDPOTrainer

logger = logging.getLogger(__name__)


def main():
    # One YAML recipe sets model, data and training options; `--key value` flags after it override them.
    parser = H4ArgumentParser((ModelArguments, DataArguments, LSCDPOConfig))
    model_args, data_args, training_args = parser.parse()
    logging.basicConfig(level=training_args.get_process_log_level(), format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)
    logger.info("Method: %s; initial beta: %s", training_args.method, training_args.beta)

    # Pairs come back chat-formatted, along with the tokenizer that formatted them.
    datasets, tokenizer = prepare_preference_datasets(model_args, data_args, training_args, logger)

    # Policy and reference both start from the SFT checkpoint; the trainer loads them as separate copies.
    trainer = LSCDPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test") if training_args.do_eval else None,
        tokenizer=tokenizer,
    )

    # A checkpoint already in output_dir means this is a restart: the optimizer, RNG state and
    # (for LSC-DPO) beta pick up where they stopped.
    checkpoint = training_args.resume_from_checkpoint or get_checkpoint(training_args)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    result.metrics["train_samples"] = len(datasets["train"])
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    # Gather the FSDP shards into one checkpoint that from_pretrained can read.
    if training_args.save_hf_model_artifacts:
        trainer.save_final_model(training_args.output_dir)

    # Held-out preference loss, only when the recipe sets do_eval.
    if training_args.do_eval:
        if "test" not in datasets:
            logger.warning("Skipping preference loss evaluation: no test split was loaded.")
        else:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(datasets["test"])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
