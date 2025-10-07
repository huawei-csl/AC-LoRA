import json
from typing import Union, Optional

import pandas as pd

template_alpaca = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:",
}


def dump_json_to_file(file: str, json_dict: dict):
    with open(file, "w+") as fp:
        json.dump(json_dict, fp=fp, indent=4)


def load_json_from_file(file: str) -> dict:
    with open(file, "r") as fp:
        json_dict = json.load(fp)
    return json_dict


"""
Code from alpaca-lora
A dedicated helper to manage templates and prompt building. To compare with LoraRetriver
"""


class Prompter(object):
    __slots__ = "template"

    def __init__(self):
        self.template = template_alpaca

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def plot_loss_from_hist(
    log_history: list[dict],
    out_file: Optional[str],
    steps: int = 10,
    plot_steps: bool = False,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    losses = pd.DataFrame(log_history)

    sns.set_style("white")
    x_axis = "step" if plot_steps else "epoch"
    labels_x = "Steps" if plot_steps else "Epoch"
    if "loss" in losses.columns:
        ax = sns.relplot(data=losses.iloc[::steps], x=x_axis, y="loss", kind="line")
        plt.xlabel(labels_x)
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.savefig(out_file)
