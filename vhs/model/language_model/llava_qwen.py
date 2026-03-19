# Modified from LLaVA: https://github.com/haotian-liu/LLaVA.git
#
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Model, Qwen2Config, Qwen2ForCausalLM, Qwen3Model

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

def is_debug():
    return int(os.environ.get('DEBUG', 0))

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"

class LlavaQwen3Config(Qwen3Config):
    model_type = "llava_qwen3"

class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

class LlavaQwen3Model(LlavaMetaModel, Qwen3Model):
    config_class = LlavaQwen3Config

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen3Model, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        if is_debug():
            n_layers = 0
            config.num_hidden_layers = n_layers
            print(f"WARNING: only {n_layers} of the model are loaded - DEBUG ONLY")
        self.model = LlavaQwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        last_cache_position = None,
        cache_position = None,
        logits_to_keep = None,
        return_loss_mask: Optional[bool] = False,
        return_image_features: Optional[bool] = False
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                return_image_features=return_image_features
            )
        if inputs_embeds is not None:
            inputs_embeds=inputs_embeds.to(self.dtype)
        loss_mask = labels !=-100
        if getattr(self.config, 'sanity_check', False):
            mask = (labels != 2152) & (labels != 9693)
            labels[mask] = -100
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep
        )
        if return_loss_mask:
            if not return_dict:
                outputs.loss_mask = loss_mask
            else:
                outputs["loss_mask"] = loss_mask
        if return_image_features:
            if not return_dict:
                outputs.image_features = image_features
            else:
                outputs["image_features"] = image_features
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=kwargs.pop("inputs_embeds"),
                **kwargs
            )
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

class LlavaQwen3ForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen3Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        if is_debug():
            n_layers = 0
            config.num_hidden_layers = n_layers
            print(f"WARNING: only {n_layers} of the model are loaded - DEBUG ONLY")
        self.model = LlavaQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        last_cache_position = None,
        cache_position = None,
        logits_to_keep = None,
        return_loss_mask: Optional[bool] = False,
        return_image_features: Optional[bool] = False
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                return_image_features=return_image_features
            )
        if inputs_embeds is not None:
            inputs_embeds=inputs_embeds.to(self.dtype)
        loss_mask = labels !=-100
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep
        )
        if return_loss_mask:
            if not return_dict:
                outputs.loss_mask = loss_mask
            else:
                outputs["loss_mask"] = loss_mask
        if return_image_features:
            if not return_dict:
                outputs.image_features = image_features
            else:
                outputs["image_features"] = image_features
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


class LlavaQwenForRegression(LlavaQwenForCausalLM):
    """LlavaQwen with a regression head (MSE loss) on top of the LLM hidden states."""

    def __init__(self, config):
        super().__init__(config)
        self.regression_head = nn.Linear(config.hidden_size, 1)
        self.regression_head.weight.data.normal_(mean=0.0, std=0.02)
        self.regression_head.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        last_cache_position=None,
        cache_position=None,
        logits_to_keep=None,
        regression_targets: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Prepare multimodal inputs (image encoding + embedding merging)
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _image_features,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.dtype)

        # Run transformer backbone to get hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Extract last non-padding token's hidden state per sequence
        if attention_mask is not None:
            # Sum attention mask to find the length, then index the last real token
            sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            sequence_lengths = sequence_lengths.clamp(min=0).long()
        else:
            sequence_lengths = torch.full(
                (hidden_states.shape[0],), hidden_states.shape[1] - 1,
                dtype=torch.long, device=hidden_states.device,
            )

        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        pooled = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden_size)

        regression_logits = self.regression_head(pooled).squeeze(-1)  # (batch,)

        loss = None
        if regression_targets is not None:
            regression_targets = regression_targets.to(regression_logits.dtype).to(regression_logits.device)
            loss = nn.functional.mse_loss(regression_logits, regression_targets)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=regression_logits.unsqueeze(-1),  # (batch, 1) for compatibility
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoConfig.register("llava_qwen3", LlavaQwen3Config)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
AutoModelForCausalLM.register(LlavaQwen3Config, LlavaQwen3ForCausalLM)
