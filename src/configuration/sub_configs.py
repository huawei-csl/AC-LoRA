from configuration.utils import PydanticBaseModelWithOptionalDefaultsPath as PBMwODP

from typing import Union, Optional, List
from pydantic import Field

import hashlib
import uuid


class DBConfig(PBMwODP, extra="forbid"):
    embedding_model_name: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="Embedding model name, needs to work with Sentence Transformer.",
    )
    db_path: Optional[str] = Field(
        None,
        description="Path to the vectorstore, if None a new db is build and saved under the out/db folder using the documents in the documents_folders.",
    )
    documents_folder: Optional[Union[str, List[str]]] = Field(
        None,
        description="All documents in this Folder/Folders (if list) will be incorporated in the vector db. Ignored if db_path is not None.",
    )
    tokens_per_chunk: int = Field(
        100,
        description="The size of chunks the datat is splitted when creating the db, if db_path not None this field is ignored.",
    )


class DataConfig(PBMwODP, extra="forbid"):
    data_dir_root: str = Field("./data", description="Root folder for the data.")
    training_data: Optional[str] = Field(
        None, description="Field ONLY used for finetuning."
    )
    evaluation_data: Optional[str] = Field(None, description="")
    img_dir: Optional[str] = Field(None, description="")


class AdapterConfig(PBMwODP, extra="forbid"):
    adapters_dir: str
    adapters: Union[List[str], str] = Field(
        "*",
        description="Either list or '*', if '*' all adapters in adapters_dir are loaded",
    )


class RetrieverParam(PBMwODP, extra="forbid"):
    k: int = Field(1, description="top-k documents retrieved")
    fetch_k: int = Field(10, description="")
    threshold: float = Field(
        0.0,
        description="LoRAs retrieved with a cosine similarity lower than this value are disregarded.",
    )


class EvalArgs(PBMwODP, extra="forbid"):
    do_sample: bool = Field(True)
    temperature: float = Field(1)
    top_k: Optional[int] = Field(50)
    num_beams: Optional[int] = Field(1)
    top_p: Optional[float] = Field(1.0)
    penalty_alpha: Optional[float] = Field(None)
    max_new_tokens: int = Field(1024)

    def get_uuid(
        self,
        repetition: int = 1,
        evaluation_data: str = "",
        db: Optional[str] = "",
        k: int = 5,
        fetch_k: int = 10,
        threshold: float = 0.0,
    ) -> str:
        if db == None:
            db = ""
        if evaluation_data == None:
            evaluation_data = ""
        seed_str = (
            self.__str__()
            + str(repetition)
            + evaluation_data
            + db
            + str(k)
            + str(fetch_k)
            + str(threshold)
        )

        m = hashlib.md5()
        m.update(seed_str.encode("utf-8"))
        return uuid.UUID(m.hexdigest()).hex


class PeftConfig(PBMwODP, extra="forbid"):
    r: int = Field(16, description="Lora rank.")
    target_modules: List[str] = Field(
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Where to add Lora",
    )
    lora_alpha: int = Field(16, description="scaling parameter, in most cases = r")
    lora_dropout: float = Field(0.0, description="unsloath optimized for 0.")
    bias: str = Field("none")
    use_rslora: bool = Field(False, description="rank stabilization")


class TrainingArgs(PBMwODP, extra="forbid"):
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int = Field(0)  # if set overwrites num_train_epochs
    num_train_epochs: int = Field(1)
    save_steps: int = Field(500)
    fp16: bool = False  # Field(not is_bfloat16_supported())
    bf16: bool = True  # Field(is_bfloat16_supported())
    logging_steps: int
    optim: str = Field("adamw_8bit")
    learning_rate: Optional[float] = Field(1e-3)
    seed: int = Field(42)

    def __str__(self):
        ret_str = ""
        do_not_consider_keys = [
            "num_train_epochs",
            "logging_steps",
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
        ]

        keys, vals = zip(*self.__dict__.items())
        # sort by keys
        svals = zip(*sorted(zip(keys, vals)))
        sval_list = list(zip(*[list(x) for x in svals]))

        for key, _ in sval_list:
            if key not in do_not_consider_keys:
                ret_str += f"{key}=[{str(getattr(self, key))}],"
        if len(ret_str) > 0:
            ret_str = ret_str[:-1]
        return ret_str
