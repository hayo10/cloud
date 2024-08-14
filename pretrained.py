import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel
from transformers.activations import ACT2FN 
from transformers import Cache, DynamicCache, StaticCache, logging, PretrainedConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import Phi3Config
from accelerate import init_empty_weights
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import PreTrainedModel
import gc
from transformers import Phi3Config
from transformers.utils import ModelOutput
from safetensors import safe_open

from model_ref import (
    _prepare_4d_causal_attention_mask_with_cache_position,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
    Phi3SuScaledRotaryEmbedding,
    Phi3YarnScaledRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
    Phi3MLP,
    repeat_kv,
    Phi3Attention,
    Phi3FlashAttention2,
    Phi3SdpaAttention,
    Phi3DecoderLayer,
    NewPhi3Config
)
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

@dataclass
class FinalOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
#     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
#     attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class Phi3PreTrainedModel(PreTrainedModel):
    config_class = Phi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PhiBody"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    
    
class Phi3Body(Phi3PreTrainedModel):
    def __init__(self, config, start, end):
        super().__init__(config)
        self.config = config

        #with init_empty_weights():
        self.body_layers = nn.ModuleList(
                [Phi3DecoderLayer(self.config, layer_idx) for layer_idx in range(start, end)]
            )

        self._attn_implementation = self.config._attn_implementation
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        embed_outputs,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
    ):
        with torch.no_grad():
            
            # hidden_states = embed_outputs[0]
            # causal_mask = embed_outputs[1]
            # position_ids = embed_outputs[2]
            # cache_positions = embed_outputs[3]
            # next_decoder_cache = embed_outputs[4]
           
            for decoder_layer in self.body_layers:

                layer_outputs = decoder_layer(
                        hidden_states = hidden_states,
                        attention_mask = embed_outputs[1],
                        position_ids = embed_outputs[2],
                        past_key_value = embed_outputs[3],
                        cache_position = embed_outputs[4],
                        next_decoder_cache = embed_outputs[5]
                    )

                hidden_states = layer_outputs[0]

            #next_decoder_cache = layer_outputs[1]
        return hidden_states
    
class Phi3Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = 32000
        #with init_empty_weights():
        self.embed_tokens = nn.Embedding(self.vocab_size+64, self.config.hidden_size)
        self.embed_dropout = nn.Dropout(self.config.embd_pdrop)
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        next_decoder_cache : Optional[torch.LongTensor] = None
    ):

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values
            )

            next_decoder_cache = None

        return (inputs_embeds, causal_mask, position_ids, past_key_values, cache_position, next_decoder_cache)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
     
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):

        if past_key_values is not None:
            if input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)


        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

#         if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:

#             batch_size, sequence_length = model_inputs["input_ids"].shape
#             device = model_inputs["input_ids"].device

#             dtype = self.lm_head.weight.dtype
#             min_dtype = torch.finfo(dtype).min

#             attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#                 attention_mask,
#                 sequence_length=sequence_length,
#                 target_length=past_key_values.get_max_length(),
#                 dtype=dtype,
#                 device=device,
#                 min_dtype=min_dtype,
#                 cache_position=cache_position,
#                 batch_size=batch_size,
#             )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
class CustomedPhi3ForCausalLM(PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, tokenizer, file_num, config: NewPhi3Config):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.config = config
        self.file_num = file_num
        # with init_empty_weights():
        self.Head_Model = Phi3Embedding(self.config)
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._attn_implementation = self.config._attn_implementation
        self.gradient_checkpointing = False

        

    def load_weights(self, partial_model, file_num, start, end, is_embedding, is_lm_head):
        """
        외장 메모리에서 decoder layer [start,end)까지 가져오기 코드
        여기에 저장하기
        """
        keys = []
        base_file_path_template = '/nas/user/hayoung/model-0000{}-of-00006.safetensors'
        base_key_name = "model.layers."
        base_body_name = "body_layers."
        included_layers = ['.input_layernorm.weight','.mlp.down_proj.weight', '.mlp.gate_up_proj.weight',
                           '.post_attention_layernorm.weight','.self_attn.o_proj.weight',
                           '.self_attn.qkv_proj.weight']

        failed_pretrained_name = []
        failed_body_name = []
        
        file_path = base_file_path_template.format(file_num)
        
            
        with safe_open(file_path, framework="pt", device="cuda") as f:
            if is_embedding:
                tensor = f.get_tensor('model.embed_tokens.weight')
                self.Head_Model.embed_tokens.weight.copy_(tensor)
            else:
                for i in range(start, end):
                    layer_name = base_key_name + str(i)
                    for name in included_layers:
                        full_name = layer_name + name
                        body_name = base_body_name + str(i-start) + name
                        if partial_model is not None:
                            # try:
                            #     tensor = f.get_tensor(full_name)
                            #     partial_model.state_dict()[body_name].copy_(tensor)
                            # except:
                            #     failed_pretrained_name.append(full_name)
                            #     failed_body_name.append(body_name)
                        
                            tensor = f.get_tensor(full_name)
                            partial_model.state_dict()[body_name].copy_(tensor)
                    
                               
            if is_lm_head:
                tensor = f.get_tensor('model.norm.weight')
                self.norm.weight.copy_(tensor)
                tensor = f.get_tensor('lm_head.weight')
                self.lm_head.weight.copy_(tensor)

        if len(failed_body_name) > 0:
        
            file_nums = []
            if file_num == 1:
                file_nums = [-1, 2]
            elif file_num == 6:
                file_nums = [5, -1]
            else:
                file_nums = [file_num - 1, file_num + 1]
            
            for num in file_nums:
                if num < 0:
                    continue
                file_path = base_file_path_template.format(num)
                with safe_open(file_path, framework="pt", device="cuda") as f:
                    for i, pre_name in enumerate(failed_pretrained_name):
                        if pre_name in f.keys():
                            tensor = f.get_tensor(pre_name)
                            partial_model.state_dict()[failed_body_name[i]].copy_(tensor)
                            failed_pretrained_name.pop(i)
                            failed_body_name.pop(i)
                
        if len(failed_pretrained_name) > 0:
            print(failed_pretrained_name)
            print(failed_body_name)
    

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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        with torch.no_grad():
            
             #next_decoder_cache = layer_outputs[1]
            
            next_decoder_cache = None
            #Head_Model = Phi3Embedding(self.config)
            self.load_weights(self.Head_Model, 1, 0, 0,is_embedding=True, is_lm_head=False)
            head_output = self.Head_Model(input_ids, attention_mask, position_ids, past_key_values, cache_position, next_decoder_cache)

            hidden_states = head_output[0]
            
            for i in range(0, 5):
                Body_Model = Phi3Body(self.config, i*5, (i+1)*5).to('cuda')
                self.load_weights(Body_Model, min(i+1,6), i*5,min((i+1)*5, 40), is_embedding=False, is_lm_head=False)
                body_output = Body_Model(hidden_states, head_output, past_key_values)
                hidden_states = body_output
                del Body_Model
                torch.cuda.empty_cache()
            
            del head_output
        
            self.load_weights(None, 6, 38, self.config.num_hidden_layers,is_embedding=False, is_lm_head=True)
            hidden_states = self.norm(body_output)

            logits = self.lm_head(hidden_states)
            logits = logits.float()
            del body_output

        return FinalOutput(logits=logits)
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):

        if past_key_values is not None:
            if input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)


        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

#         if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:

#             batch_size, sequence_length = model_inputs["input_ids"].shape
#             device = model_inputs["input_ids"].device

#             dtype = self.lm_head.weight.dtype
#             min_dtype = torch.finfo(dtype).min

#             attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#                 attention_mask,
#                 sequence_length=sequence_length,
#                 target_length=past_key_values.get_max_length(),
#                 dtype=dtype,
#                 device=device,
#                 min_dtype=min_dtype,
#                 cache_position=cache_position,
#                 batch_size=batch_size,
#             )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    