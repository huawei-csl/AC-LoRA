# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Union, List
from peft.peft_model import PeftModel

import torch
from peft.utils import PeftType, _set_adapter


class PeftModelMoLE(PeftModel):
    """Code modified from peft library"""

    @property
    def active_peft_config(self):
        if list(self.peft_config.keys()):
            # @ACLoRA ALL LORAS need to have the same config ATM
            return self.peft_config[list(self.peft_config.keys())[0]]
        else:
            return self.peft_config

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        if self.active_peft_config == {}:  # required if no LoRA is retrieved.
            return self.base_model.model

        return (
            self.base_model
            if (
                self.active_peft_config.is_prompt_learning
                or self.peft_type == PeftType.POLY
            )
            else self.base_model.model
        )

    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """
        Sets the active adapter.

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
        if type(adapter_name) == str:
            adapter_name = [adapter_name]

        if any(adapter not in self.peft_config for adapter in adapter_name):
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        self.base_model.set_adapter(
            [
                name
                for name in adapter_name
                if not self.peft_config[name].is_prompt_learning
            ]
        )
        _set_adapter(self, adapter_name)
