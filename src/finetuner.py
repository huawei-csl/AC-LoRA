from unsloth import FastLanguageModel, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

from configuration import Config
from utils import plot_loss_from_hist
from pathlib import Path

import logging
from glob import glob


class Finetuner:
    def __init__(self, config: Config):
        if not config.model.peft_config:
            logging.error("Cannot finetune a LoRA without peft config!")
            return

        if config.model.lora_training_args:
            training_args = dict(config.model.lora_training_args)
        else:
            training_args = {}
        self.model_type = config.model.model_type
        if config.model.model_type == "multitit":
            # for Qwen2-vl
            model, self.tokenizer = FastVisionModel.from_pretrained(
                config.model.base_model,
                load_in_4bit=False,
                use_gradient_checkpointing="unsloth",
                cache_dir=config.meta.cache_dir,
            )
            self.data_collator = UnslothVisionDataCollator(model, self.tokenizer)
            self.peft_model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                **dict(config.model.peft_config),
            )
            # additional args required for Vision FT
            training_args = {
                **training_args,
                "remove_unused_columns": False,
                "dataset_text_field": "",
                "dataset_kwargs": {"skip_prepare_dataset": True},
                "dataset_num_proc": 4,
                "max_seq_length": 2048,
            }
            self.img_dir = config.data.img_dir
        elif config.model.model_type == "text":
            model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model.base_model,
                load_in_4bit=False,
                cache_dir=config.meta.cache_dir,
            )
            self.data_collator = None
            self.peft_model = FastLanguageModel.get_peft_model(
                model, **dict(config.model.peft_config)
            )
            FastLanguageModel.for_training(self.peft_model)
            training_args = {**training_args}  # only for Instruct

        else:
            logging.error(
                "Stable Diffusion finetuning " + "is not handled from this code."
            )
            return

        # Multiple files can be Loaded either with glob or by separate them with a ','
        self.data_files = [
            x
            for xs in [
                glob(config.data.data_dir_root + file)
                for file in config.data.training_data.split(",")
            ]
            for x in xs
        ]

        self.training_args = training_args
        self.seed = config.meta.seed
        self.result_dir = config.meta.out_root_dir + config.meta.result_dir
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

    def convert_mmsci_to_conversation(self, example):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["conversations"][0]["value"]},
                    {"type": "image", "image": self.img_dir + example["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["conversations"][1]["value"]}
                ],
            },
        ]
        return {"messages": conversation}

    def get_formatting_fun(self, example_datapoint):
        # in utils
        if "messages" in example_datapoint.keys():

            def formatting_prompts_func(example):
                # unsloth expects text as format
                texts = self.tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
                return {"text": texts}

            return formatting_prompts_func
        elif "image" in example_datapoint.keys():
            # placeholder because unsloth requires weird format
            def id(x):
                return x

            return id
        else:
            # TODO for other dataset might require different formatting functions
            logging.error("Formatting Function non implemented")
            pass

    def train(self, plot_loss: bool = True):
        dataset = load_dataset("json", data_files=self.data_files, split="train")
        dataset = dataset.shuffle(seed=self.seed)
        formatting_fun = self.get_formatting_fun(dataset[0])
        dataset = dataset.map(formatting_fun)
        if self.model_type == "multitit":
            dataset = [self.convert_mmsci_to_conversation(x) for x in dataset]

        trainer = SFTTrainer(
            model=self.peft_model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=SFTConfig(output_dir=self.result_dir + "/out", **self.training_args),
        )

        trainer.train()

        trainer.save_model(output_dir=self.result_dir + "/model")
        if plot_loss:
            plot_loss_from_hist(
                trainer.state.log_history,
                self.result_dir + "/out/loss_plot.png",
                steps=1,
                plot_steps=True,
            )
