from configuration.utils import PydanticBaseModelWithOptionalDefaultsPath as PBMwODP
from configuration.sub_configs import (
    DataConfig,
    TrainingArgs,
    RetrieverParam,
    AdapterConfig,
    EvalArgs,
    DBConfig,
    PeftConfig,
)
from typing import Optional
from pydantic import Field

import hashlib
import os
import uuid


class RetrieverConfig(PBMwODP, extra="forbid"):
    db: DBConfig
    params: RetrieverParam
    permissions: str = Field(
        "*",
        description="List of permitted LoRA as string divided by a ','. '*' if all LoRAs are permitted.",
    )
    no_prediction: bool = Field(
        False, description="if True, ACLoRA returns after retrieving the LoRAs."
    )
    hinting: bool = Field(False, description="")


class EvalConfig(PBMwODP, extra="forbid"):
    eval_args: EvalArgs = Field(..., description="")
    repetitions_eval: int = Field(1, description="")


class GradingConfig(PBMwODP, extra="forbid"):
    type: str = Field(
        "grading", description="Evaluation type. Options: [grading|grading_api]"
    )
    ollama: bool = Field(
        True,
        description="If grading it using ollama or not. Ignored if evaluation_type is not grading.",
    )
    grading_model: str = Field(
        "gemma3:27b", description="Judging model. Only used for type 'grading'"
    )
    repetitions: int = Field(1, description="How often we prompt the judge.")
    temperature: float = Field(
        0.1, description="Temperature given to ollama for grading."
    )
    url: str = Field(
        "https://api.deepinfra.com/v1/openai",
        description="Url for api endpoint if type is api_grading",
    )


class ModelConfig(PBMwODP, extra="forbid"):
    base_model: str = Field(..., description="Model name, either path or hf url.")
    model_type: str = Field("text", description="Options=[text,multitit,multisd]")
    peft_config: Optional[PeftConfig] = Field(
        None,
        description="Peft configuration of the LoRAs to beloaded, if None runs base model",
    )
    adapters: Optional[AdapterConfig] = Field(
        "default", description="If retrieverconfig None, eval with all in list on avg."
    )
    lora_training_args: Optional[TrainingArgs] = Field(
        None, description="Training arguments"
    )


class MetaConfig(PBMwODP, extra="forbid"):
    out_root_dir: str = Field(..., description="Directory to save outputs to")
    result_dir: str = Field(
        "results/", description="Directory to save evaluation results to"
    )
    cache_dir: str = Field(".cache/huggingface", description="Torch cache")
    seed: int = Field(42, description="Random seed")

    def short_str(self) -> str:
        return f"seed={self.seed}"


class Config(PBMwODP, extra="forbid"):
    meta: MetaConfig = Field(..., description="")
    model: Optional[ModelConfig] = Field(None, description="")
    data: DataConfig = Field(..., description="")
    retriever: Optional[RetrieverConfig] = Field(None, description="")
    evaluation: Optional[EvalConfig] = Field(None, description="")
    grading: Optional[GradingConfig] = Field(None, description="")

    def get_result_path(self) -> str:
        return str(os.path.join(self.meta.out_root_dir, self.meta.result_dir))

    def get_uuid(self) -> str:
        seed_str = self.__str__()

        m = hashlib.md5()
        m.update(seed_str.encode("utf-8"))
        return uuid.UUID(m.hexdigest()).hex
