from configuration import Config, RetrieverConfig, AdapterConfig
from models import TextModel, MultiModalTIT, MultiModalSD
from utils import dump_json_to_file

from peft_patch.peft_model import PeftModelMoLE
from peft_patch.lora_fixed_gate import Linear
from unittest.mock import patch

from langchain_community.vectorstores import FAISS
from datasets import load_dataset
import torch

from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import numpy as np
from glob import glob
import logging

from PIL.Image import Image


# Patching the peft Library to enable Mixing
patch("peft.peft_model.PeftModel.set_adapter", PeftModelMoLE.set_adapter).start()
patch("peft.peft_model.PeftModel.get_base_model", PeftModelMoLE.get_base_model).start()
patch(
    "peft.peft_model.PeftModel.active_peft_config", PeftModelMoLE.active_peft_config
).start()
patch("peft.tuners.lora.layer.Linear", Linear).start()


class ACLoRA:
    """
    Implementation of ACLoRA - containing the retrieving and end-to-end generation given a config file.
    """

    def __init__(self, config: Config, load_loras_at_init: bool = True):
        self.permission = None
        if config.retriever:
            db_config = config.retriever.db
            db_path = db_config.db_path
            self.retriever_params = dict(config.retriever.params)
            self.no_prediction = config.retriever.no_prediction
            self.permission = (
                config.retriever.permissions
                if (
                    config.retriever.permissions == "*"
                    or config.retriever.permissions == "hint-test"
                )
                else config.retriever.permissions.split(",")
            )
            self.hinting_enabled = config.retriever.hinting
        else:
            db_config = None
            db_path = ""
            self.no_prediction = False
            self.retriever_params = {}

        if config.model.model_type == "multitit":
            self.models = MultiModalTIT(
                model_config=config.model,
                cache_dir=config.meta.cache_dir,
                db=db_config,
                eval_config=config.evaluation,
            )
        elif config.model.model_type == "multisd":
            self.models = MultiModalSD(
                model_config=config.model,
                cache_dir=config.meta.cache_dir,
                db=db_config,
                eval_config=config.evaluation,
                seed=config.meta.seed,
            )
        else:
            self.models = TextModel(
                model_config=config.model,
                cache_dir=config.meta.cache_dir,
                db=db_config,
                eval_config=config.evaluation,
            )

        # Multiple files can be Loaded either with glob or by separate them with a ','
        self.data_files = [
            x
            for xs in [
                glob(config.data.data_dir_root + file)
                for file in config.data.evaluation_data.split(",")
            ]
            for x in xs
        ]

        # name of eval file
        if config.evaluation:
            self.uuid = config.evaluation.eval_args.get_uuid(
                repetition=config.evaluation.repetitions_eval,
                evaluation_data="".join(self.data_files),
                db=db_path,
                **self.retriever_params,
            )
        else:
            self.uuid = config.get_uuid()

        self.result_dir = config.meta.out_root_dir + config.meta.result_dir
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.img_dir = config.data.img_dir
        self.db = None
        self.loras = {}
        if config.model.peft_config:
            self.setup_retriever(
                config=config.retriever,
                lora_config=config.model.adapters,
                load_loras_at_init=load_loras_at_init,
            )
        else:
            # if no peft - we run base model
            logging.info("No PEFT config in configuration. Only running Base Model")

    def load_loras(self, lora_config: AdapterConfig, load_loras_at_init: bool):
        """
        Create self.lora dict from lora name to lora path. Loads all lora to
        device if load_loras_at_init.
        """
        self.loras = {}

        adapters = glob(lora_config.adapters_dir + lora_config.adapters)
        for adapter in adapters:
            # It expects the name of the Directory to be the name of the adapter
            self.loras[adapter.split("/")[-1]] = adapter

        if load_loras_at_init:
            self.models.load_all_loras(self.loras)

    def setup_retriever(
        self,
        config: RetrieverConfig,
        lora_config: AdapterConfig,
        load_loras_at_init: bool,
    ):
        """
        Setups the Retriever: load vector database from storage or if no path exists it builds
        the vector base from scratch using the files given in config.documents_folder and saves
        it at result_dir+/db.
        Loads LoRAs to device if load_loras_at_init is True.
        Returns False if failed to setup the retriever, True otherwise.
        """

        if config:
            # return False
            if config.db.db_path:
                self.db = FAISS.load_local(
                    config.db.db_path,
                    self.models.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            else:
                self.db = self.models.build_db(
                    db=config.db, output_path=f"{self.result_dir}/db"
                )
        else:
            logging.info("No retriever configuration found. Disabling ACLoRa.")

        self.load_loras(lora_config=lora_config, load_loras_at_init=load_loras_at_init)

    def _get_top_loras_text(self, prompt, permission):
        if permission == "*":
            vec_db_response = self.db.similarity_search_with_score(
                query=prompt, **self.retriever_params
            )
        else:
            vec_db_response = self.db.similarity_search_with_score(
                query=prompt,
                **self.retriever_params,
                filter={"source": {"$in": permission}},
            )
        return vec_db_response

    def _get_top_loras_ti(self, prompt, permission):
        prompt_embedding = self.models.embedding_model.encode(prompt).tolist()[0]
        if permission == "*":
            vec_db_response = self.db.similarity_search_with_score_by_vector(
                embedding=prompt_embedding, **self.retriever_params
            )
        else:
            vec_db_response = self.db.similarity_search_with_score_by_vector(
                embedding=prompt_embedding,
                **self.retriever_param,
                filter={"source": {"$in": permission}},
            )
        return vec_db_response

    def _process_db_response(self, db_response: dict):
        """
        Processes the vector response and returns a list of LoRAs and their corresponsing
        gate values.
        """
        ## TODO can directly at to the similarity score I guess
        threshold = self.retriever_params["threshold"]
        weights_l = {}

        for r in db_response:
            if "source" in r[0].metadata.keys():
                lora = r[0].metadata["source"]
            else:
                lora = r[0].page_content
            weights_l.setdefault(lora, []).append(r[1])

        avg_weights = {k: np.mean(v) for k, v in weights_l.items()}
        filtered = [(k, v) for k, v in avg_weights.items() if v > threshold]

        if filtered:
            rel_loras, rel_scores = zip(*filtered)
            rel_weights = torch.nn.Softmax(dim=0)(
                torch.tensor(rel_scores, dtype=torch.float32)
            )
            return list(rel_loras), rel_weights
        return [], []

    def _get_top_loras_fun(self):
        return (
            self._get_top_loras_ti
            if isinstance(self.models, MultiModalTIT)
            else self._get_top_loras_text
        )

    def get_top_loras(
        self, prompt: str, permission: Union[str, List] = "*", hint: bool = False
    ):
        _get_top_loras = self._get_top_loras_fun()
        hint_lora, hint_weight = None, None
        if hint and permission != "*":
            # call once with all permissions if hinting is enabled and not all permissions
            hint_db_response = _get_top_loras(prompt, "*")
            hint_loras, hint_weights = self._process_db_response(
                db_response=hint_db_response
            )
            max_hint_arg = torch.argmax(hint_weights)
            hint_lora, hint_weight = (
                hint_loras[max_hint_arg],
                hint_weights[max_hint_arg],
            )
        db_response = _get_top_loras(prompt=prompt, permission=permission)
        loras, gate = self._process_db_response(db_response=db_response)
        # small fix because added . in db.prob not needed later
        loras = [
            l.replace(".", "").replace("jsonl", "").replace("json", "") for l in loras
        ]
        return loras, gate, hint_lora, hint_weight

    def _evaluate(self, datapoint: Union[dict, str]):
        """
        Eval of one prompt. If no_prediction is true it returns after retrieving
        the LoRAs.
        """
        loras, gate_weights = [], torch.tensor([])
        str_prompt = self.models.extract_prompt(datapoint, self.img_dir)
        if self.db:
            # ACLoRA
            loras, gate_weights, hint_lora, hint_weight = self.get_top_loras(
                prompt=str_prompt, permission=self.permission, hint=self.hinting_enabled
            )
        elif self.loras:
            # Basic Avg of all Listed LoRAs
            loras = self.loras
            gate_weights = torch.tensor([1 / len(loras)] * len(loras))

        if self.loras:
            if self.no_prediction:
                return "", loras, gate_weights

            self.models.load_and_set_selected_loras(
                retrieved_loras=loras, gate=gate_weights, loras=self.loras
            )

        answers = self.models._generate(datapoint, str_prompt)

        answers_hint = ""
        loras_full = []
        if self.loras:
            if self.test_hint and hint_lora and (hint_lora not in loras):
                loras_full, gate_weights_full, _, _ = self.get_top_loras(
                    prompt=str_prompt, permission="*", hint=False
                )

                self.models.load_and_set_selected_loras(
                    retrieved_loras=loras_full, gate=gate_weights_full, loras=self.loras
                )

                answers_hint = self.models._generate(datapoint, str_prompt)

        return answers, loras, gate_weights, answers_hint, loras_full

    def evaluate(self):
        """
        Evaluate all files in self.data_files. Check files loaded (data/*/test) to see which formats are supported.
        Other might require overwriting the format function or adapting the file to one of the supported formats.

        Writes results to a single json file.
        """
        test_sets = [
            (
                file.split("/")[-1].replace("_test", "").replace(".json", ""),
                load_dataset("json", data_files={"test": file}, split="test"),
            )
            for file in self.data_files
        ]
        self.test_hint = False
        if self.permission == "hint-test":
            self.test_hint = True
        output = [dict(self.retriever_params)]
        for tag, test_set in test_sets:
            if self.test_hint:
                self.permission = [lora for lora in self.loras.keys() if lora != tag]
            # test_set = test_set.map(self.models.format)
            for idx, data_point in enumerate(tqdm(test_set, desc=f"Evaluating {tag}")):
                predictions, retrieved_loras, gate_loras, hint_prediction, hint_lora = (
                    self._evaluate(data_point)
                )
                if type(predictions) == Image:
                    predictions.save(f"{self.result_dir}/{self.uuid}_{idx}.png")
                    output.append(
                        {
                            "tag": tag,
                            "idx": idx,
                            **data_point,
                            "retrieved_loras": retrieved_loras,
                            "gate": gate_loras.tolist(),
                        }
                    )
                    pass
                else:
                    if type(gate_loras) != list:
                        gl = gate_loras.tolist()
                    else:
                        gl = gate_loras
                    output.append(
                        {
                            "tag": tag,
                            **data_point,
                            "prediction": predictions,
                            "hint_prediction": hint_prediction,
                            "hint_lora": hint_lora,
                            "retrieved_loras": retrieved_loras,
                            "gate": gl,
                        }
                    )

            dump_json_to_file(f"{self.result_dir}/{self.uuid}.json", output)
