from configuration import ModelConfig, DBConfig, EvalConfig
from typing import Optional, List
from peft import get_peft_model, LoraConfig
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
import torch

from tqdm import tqdm
from glob import glob

from utils import Prompter
from diffusers import AutoPipelineForText2Image

from qwen_vl_utils import process_vision_info


class ModelType:
    """Base class"""

    def __init__(self, model_config: ModelConfig, eval_config: EvalConfig):
        self.model_type = model_config.model_type

        self.embedding_model = None
        self.peft_model = None
        self.tokenizer = None
        self.all_loras_loaded = False
        self.prompter = None  # only used for FLAN - same as LoRARetriver

        self.generation_kwargs = {}
        self.repetitions = 1
        if eval_config:
            self.generation_kwargs = dict(eval_config.eval_args)
            self.repetitions = eval_config.repetitions_eval

    def format(prompt):
        pass

    def get_formatting_function():
        return format

    def format(self, prompt: dict):
        return self.get_formatting_function()(prompt)

    def load_all_loras(self, loras: dict):
        """ """
        self.all_loras_loaded = True
        for adapter, adapter_path in tqdm(loras.items(), desc="Loading LoRAs"):
            self.peft_model.load_adapter(adapter_path, adapter_name=adapter)

    def load_and_set_selected_loras(
        self, retrieved_loras: List[str], gate: torch.Tensor, loras: dict
    ):
        """ """
        retrieved_loras = set(retrieved_loras)
        if not self.all_loras_loaded:
            # if for some reason we did not load all LoRAs at the beginning - memory constrained. We only
            # load the one we at the moment need. This requires less memory but decreases performance.
            loaded_adapters = set(
                [self.peft_model.active_adapter]
                if isinstance(self.peft_model.active_adapter, str)
                else self.peft_model.active_adapter
            )
            to_remove = loaded_adapters - retrieved_loras
            to_add = retrieved_loras - loaded_adapters

            for adapter in to_remove:
                self.peft_model.delete_adapter(adapter)

            for adapter in to_add:
                self.peft_model.load_adapter(loras[adapter], adapter_name=adapter)

        self.set_adapter(list(retrieved_loras))
        if len(retrieved_loras) > 1:
            self.set_gate(weights=gate)

    def set_adapter(self, loras):
        self.peft_model.set_adapter(loras)

    def build_db(self, db: DBConfig, output_path: str):
        """Given ONLY text embeddings this function incorporates the given documents
        and return a FAISS vector store"""
        file_texts = {}
        docs = glob(db.documents_folder)
        for d in docs:
            # Expected format is that the name of the LoRA is the same as the file, might have added _train.json (or jsonl) or not.
            lora_tag = (
                d.split("/")[-1]
                .replace("_train", "")
                .replace(".jsonl", "")
                .replace(".json", "")
            )
            with open(d, "r") as f:
                file_texts[(d, lora_tag)] = f.read()

        docs = {}
        text_splitter = SentenceTransformersTokenTextSplitter(
            model_name=db.embedding_model_name, tokens_per_chunk=db.tokens_per_chunk
        )
        for k, v in tqdm(file_texts.items(), desc="Creating documents"):
            docs[k] = text_splitter.create_documents([v])

        documents = []
        for (_, tag), v in tqdm(docs.items(), desc="Processing documents"):
            for doc in v:
                documents.append(
                    Document(page_content=doc.page_content, metadata={"source": tag})
                )

        db = None
        for d in tqdm(documents, desc="Ingesting documents"):
            if db:
                db.add_documents([d])
            else:
                db = FAISS.from_documents([d], self.embedding_model)
        print(f"Save db to {output_path}")
        db.save_local(output_path)
        return db

    def set_gate(self, weights: torch.Tensor):
        for m in self.peft_model.modules():
            if hasattr(m, "gate"):
                m.update_gate(weights)


class MultiModalTIT(ModelType):
    """Model class for the Qwen2-VL (text-image to text) model."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_dir: str,
        db: Optional[DBConfig] = None,
        eval_config: Optional[EvalConfig] = None,
    ):
        super().__init__(model_config, eval_config)

        if db:
            self.embedding_model = SentenceTransformer(
                db.embedding_model_name,
                trust_remote_code=True,
                device="cuda",
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "sdpa",
                },
                config_kwargs={"is_text_encoder": False, "vector_dim": 1024},
                cache_folder=cache_dir,
            )
            self.embedding_model.max_seq_length = 1024

        self.peft_model = AutoModelForImageTextToText.from_pretrained(
            model_config.base_model, device_map="auto", cache_dir=cache_dir
        )

        if model_config.peft_config:
            # If we have a peft config. Otherwise we just run the base model
            self.peft_model = get_peft_model(
                self.peft_model, LoraConfig(**dict(model_config.peft_config))
            )

        self.tokenizer = AutoProcessor.from_pretrained(
            model_config.base_model, cache_dir=cache_dir, padding_side="right"
        )

    def extract_prompt(self, dp: dict, img_dir: str) -> dict:
        # For Qwen2-VL + jasper - MMSci dataset, for anything else one might need to adapter
        # the output
        if img_dir:
            return [
                [
                    {"type": "text", "content": dp["question"]},
                    {"type": "image_path", "content": f"{img_dir}/{dp['image']}"},
                ]
            ]
        else:
            return []

    def build_db(self, db: DBConfig, output_path: str):
        # Needs overwriting
        return

    def _generate(self, dp: dict, str_prompt: str) -> List[str]:
        """
        Make one generate and get formatting function.
        """
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str_prompt[0][1]["content"],
                    },
                    {"type": "text", "text": str_prompt[0][0]["content"]},
                ],
            }
        ]

        input_text = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(prompt)
        ids = self.tokenizer(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        answers = []
        for _ in range(self.repetitions):
            with torch.no_grad():
                output = self.peft_model.generate(**ids, **self.generation_kwargs)
            answers.append(
                self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].split(
                    "\nassistant\n"
                )[-1]
            )
        return answers


class MultiModalSD(ModelType):
    """For stable diffusion (text-to-image) models"""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_dir: str,
        db: Optional[DBConfig] = None,
        eval_config: Optional[EvalConfig] = None,
        seed: int = 42,
    ):
        super().__init__(model_config, eval_config)

        if db:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=db.embedding_model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": False},
            )
        self.seed = seed
        self.peft_model = AutoPipelineForText2Image.from_pretrained(
            model_config.base_model, torch_dtype=torch.float16, cache_dir=cache_dir
        ).to("cuda")

    def load_all_loras(self, loras: dict):
        self.all_loras_loaded = True
        # adapters need to be sorted otherwise diffusers library is confused
        adapters = sorted(loras.keys(), key=lambda x: len(x.split("_")))
        for adapter in tqdm(adapters, desc="Loading LoRAs"):
            if (
                not hasattr(self.peft_model.unet, "peft_config")
                or adapter not in self.peft_model.unet.peft_config.keys()
            ):
                self.peft_model.load_lora_weights(
                    loras[adapter],
                    weight_name="pytorch_lora_weights.safetensors",
                    adapter_name=adapter,
                )

    def set_adapter(self, loras):
        # The results using this to set weights somehow does not work
        # even without out patching
        self.peft_model.set_adapters(loras)

    def set_gate(self, weights: torch.Tensor):
        for m in self.peft_model.unet.modules():
            if hasattr(m, "gate"):
                m.update_gate(weights)

    def extract_prompt(self, dp: dict, args):
        # example prompts for sd
        return dp["input"]

    def _generate(self, dp: dict, str_prompt: str):
        generator = torch.Generator("cuda").manual_seed(self.seed)
        return self.peft_model(str_prompt, generator=generator).images[0]


class TextModel(ModelType):
    """For text models (explicitly for our experiments of the FLan and RepLiQA dataset.)"""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_dir: str,
        db: Optional[DBConfig] = None,
        eval_config: Optional[EvalConfig] = None,
    ):
        super().__init__(model_config, eval_config)

        if db:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=db.embedding_model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": False},
            )

        self.peft_model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model, device_map="auto", cache_dir=cache_dir
        )

        if model_config.peft_config:
            self.peft_model = get_peft_model(
                self.peft_model, LoraConfig(**dict(model_config.peft_config))
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.base_model, cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.generation_kwargs:
            self.generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

    def extract_prompt(self, dp: dict, agrs) -> str:
        # We expect here the data to have either the format as in our RepLiQA files or
        # Flan test files, if other format is given, this function needs to be adapted.
        if "inputs" in dp.keys() and not self.prompter:
            # FLANV2
            self.prompter = Prompter()
        if "inputs" in dp.keys() and self.prompter:
            return self.prompter.generate_prompt(
                dp["inputs"],
                "",
                "",
            )
        else:
            # RepliQA
            return dp["messages"][0]["content"]

    def _generate(self, dp: dict, str_prompt: str) -> List[str]:
        """ """
        # tokenize depending on dataset
        if "inputs" in dp.keys() and self.prompter:
            # Flanv2
            ids = self.tokenizer(
                str_prompt,
                max_length=512,
                return_tensors="pt",
                # padding=True,
            ).to("cuda")
        else:
            # RepliQA
            ids = self.tokenizer.apply_chat_template(
                dp["messages"],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
        answers = []
        for _ in range(self.repetitions):
            with torch.no_grad():
                output = self.peft_model.generate(**ids, **self.generation_kwargs)
            answers.append(
                self.tokenizer.batch_decode(output, skip_special_tokens=True)
            )
        return answers
