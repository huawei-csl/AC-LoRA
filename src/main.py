import argparse

from configuration.meta_config import get_pydantic_models_from_path

from configuration.config import Config

import gc
import torch

import numpy as np
import random


def fix_randomenss(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def parse_configs() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        nargs="+",
        help="Path to config file to run. Could be more than one.",
    )
    parser.add_argument(
        "--load_loras",
        action="store_true",
        help="If all loaded should be loaded at init or not.",
    )
    arguments = parser.parse_args()

    configs = []
    for config_path in arguments.config:
        configs += get_pydantic_models_from_path(config_path)

    return arguments.load_loras, configs


def main():
    """
    python main.py --config config1.yaml cofig2.yaml --load_loras
    configs are executed in order.
    """
    load_loras, configs = parse_configs()
    for config in configs:
        # fix randomness to seed given in config,
        # to make results as reproducible as possible
        fix_randomenss(config.meta.seed)
        #
        if config.model and config.model.lora_training_args:
            # For finetuning text and multi
            # ensure unlosth is only Loaded when finetuning
            from finetuner import Finetuner

            Finetuner(config).train()
        elif config.grading:
            from grading import Grader

            Grader(config).grade()
        else:
            # ensuring we only patch when not finetuning
            from aclora import ACLoRA

            ACLoRA(config, load_loras).evaluate()

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
