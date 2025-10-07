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

import warnings
from typing import Any, Optional, Union

import torch

from peft.import_utils import is_bnb_4bit_available
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight, gather_params_ctx
from peft.utils.other import transpose

import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora.layer import LoraLayer

from torch.distributions import Categorical

import math

"""
ACLoRA - Implementation from https://github.com/huggingface/peft/tree/main/src/peft/tuners/lora
to patch to enable mixing. Main changes in the forward function. Changes marked with @ACLoRA
"""


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        # @ACLoRA : initializing the gate function.
        self.gate = torch.tensor(
            0,
            requires_grad=False,
            device=next(self.get_base_layer().parameters()).device,
        )

    # @ACLoRA: Set values of the gate.
    def update_gate(self, weights: torch.Tensor):
        layer = next(self.get_base_layer().parameters())
        self.gate = weights.to(layer.device, dtype=layer.dtype)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                orig_weights,
                                transpose(delta_weight, self.fan_in_fan_out),
                                scaling=1,
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = (
                            self.lora_magnitude_vector[active_adapter].weight
                            / weight_norm
                        )
                        dora_factor = transpose(
                            dora_factor.view(-1, 1), self.fan_in_fan_out
                        )
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight,
                                transpose(delta_weight, self.fan_in_fan_out),
                                scaling=1,
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = (
                            self.lora_magnitude_vector[active_adapter].weight
                            / weight_norm
                        )
                        dora_factor = transpose(
                            dora_factor.view(-1, 1), self.fan_in_fan_out
                        )
                        new_weight = dora_factor * (
                            base_layer.weight.data + delta_weight
                        )
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = (
                        self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    )
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16
        )

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (
            transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            results_single_loras = []
            # torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                # x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    output = lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    output = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )
                # @ACLoRA: Save different LoRA outputs
                results_single_loras.append(output)

            # @ACLoRA: If more than one LoRA multiply with gate, ow just use normal LoRA result
            if len(self.active_adapters) > 1:
                loras = torch.stack(results_single_loras, dim=2)
                results_loras = torch.einsum("blnd,n->bld", loras, self.gate)

            elif len(self.active_adapters) == 1:
                results_loras = output

            if len(self.active_adapters) > 0:
                result = result + results_loras

            # result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


if is_bnb_4bit_available():
    import bitsandbytes as bnb

    # @ACLoRA same changes as for the full precision above.
    class Linear4bit_gate(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self.entropy = 0
            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
            )
            self.gate = torch.tensor(
                0,
                requires_grad=False,
                device=next(self.get_base_layer().parameters()).device,
            )

        def update_gate(self, weights: torch.Tensor):
            self.gate = weights.to(next(self.get_base_layer().parameters()).device)

        def merge(
            self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
        ) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue

                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                lora_data = self.get_delta_weight(active_adapter)

                output = dequantize_bnb_weight(weight, state=weight.quant_state)
                if not self.use_dora[active_adapter]:
                    w_data = output + lora_data
                else:
                    # handle dora
                    # since output already includes scaling, set it to 1 here
                    weight_norm = (
                        self.lora_magnitude_vector[active_adapter]
                        .get_weight_norm(output, lora_data, scaling=1)
                        .detach()
                    )
                    # We need to cache weight_norm because it has to be based on the original weights. We
                    # cannot calculate it on the fly based on the merged weights when unmerging because its a
                    # different value
                    self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                    dora_factor = (
                        self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    )
                    w_data = dora_factor.view(-1, 1) * (output + lora_data)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.nn.Params4bit(
                    w_data.to("cpu"), **kwargs
                ).to(weight.device)
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )

                lora_data = self.get_delta_weight(active_adapter)
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                output = dequantize_bnb_weight(weight, state=weight.quant_state)

                if not self.use_dora[active_adapter]:
                    w_data = output - lora_data
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = (
                        self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    )
                    w_data = output.data / dora_factor.view(-1, 1) - lora_data

                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.nn.Params4bit(
                    w_data.to("cpu"), **kwargs
                ).to(weight.device)

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append(
                    [
                        index
                        for index, item in enumerate(adapter_names)
                        if item == adapter
                    ]
                )

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)
                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                output = lora_B(lora_A(dropout(sub_batch))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result[sub_batch_indices_list[i]] += output

            return result

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(
                    x, *args, adapter_names=adapter_names, **kwargs
                )
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()
                results_single_loras = []
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = x.to(lora_A.weight.dtype)

                    if not self.use_dora[active_adapter]:
                        output = lora_B(lora_A(dropout(x))) * scaling
                    else:
                        x = dropout(x)
                        output = self.lora_magnitude_vector[active_adapter](
                            x,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            scaling=scaling,
                            base_layer=self.get_base_layer(),
                        )
                    results_single_loras.append(output)

                if len(self.active_adapters) > 1:
                    loras = torch.stack(results_single_loras, dim=2)
                    results_loras = torch.einsum("blnd,n->bld", loras, self.gate)

                elif len(self.active_adapters) == 1:
                    results_loras = output
                if len(self.active_adapters):
                    if requires_conversion:
                        results_loras = results_loras.to(expected_dtype)

                    result = result + results_loras

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep
