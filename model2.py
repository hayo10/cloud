from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import PreTrainedModel, Phi3Config
from transformers.utils import ModelOutput
from safetensors import safe_open
import json
from hf_ref import (
    _prepare_4d_causal_attention_mask_with_cache_position,
    Phi3RMSNorm,
    Phi3DecoderLayer,
    NewPhi3Config
)
from transformers.cache_utils import StaticCache

device = torch.device('cuda:0')
torch.cuda.set_device(device) 
pre_weight_map = {}
file_num = 1
tensor_dict = {}
    
    

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None

class Phi3PreTrainedModel(PreTrainedModel):
    config_class = NewPhi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    _version = "0.0.5"

                
class EmbedModel(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
 
        global device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx).to(device)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)   
        
    def load_weights(self):
        global pre_weight_map, tensor_dict
        
        new_state_dict = {}
        new_state_dict['embed_tokens.weight'] = tensor_dict['model.embed_tokens.weight']
        
        self.load_state_dict(new_state_dict)
        del tensor_dict['model.embed_tokens.weight']
    
    def forward(
        self,
        input_ids: torch.LongTensor = None
    ):
        self.load_weights()
        inputs_embeds = self.embed_tokens(input_ids)

        return inputs_embeds
        
        
        
        
class Body(Phi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]

    Args:
        config: Phi3Config
    """

    def __init__(self, block_size, config: NewPhi3Config):
        super().__init__(config)
        global device
        self.layers = nn.ModuleList(
            [Phi3DecoderLayer(config, i) for i in range(block_size)]
        ).to(device)
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.block_size = block_size
        self.config = config

    def load_one_file(self):
    
        global file_num, tensor_dict
        
        if file_num > 6:
            print("파일 번호가 6번을 넘어감")
        file_path = self.config.base_path + f'/model-0000{file_num}-of-00006.safetensors'
        
        with safe_open(file_path, framework="pt", device="cuda") as f:
            for key in f.keys():
                tensor_dict[key] = f.get_tensor(key)
        
        file_num += 1

    def load_weights(self, idx):
        
        new_state_dict = {}
        global pre_weight_map, tensor_dict
        
        partial_model_keys = list(self.layers.state_dict().keys())
        
        names = []
        for key in partial_model_keys:
            try:
                num = int(key[0:2])
                pre_num = num + idx
                pre_name = "model.layers." + str(pre_num) + key[2:]
            except:
                num = int(key[0])
                pre_num = num + idx
                pre_name = "model.layers." + str(pre_num) + key[1:]
            try:
                new_state_dict[key] = tensor_dict[pre_name]
                names.append(pre_name)
            except:
                self.load_one_file()
                new_state_dict[key] = tensor_dict[pre_name]
        
        
        self.layers.load_state_dict(new_state_dict)
        
        for name in names:
            del tensor_dict[name]
        del partial_model_keys
        
        
        


    def forward(
        self,
        idx,
        hidden_states: torch.LongTensor = None,
        causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        
        self.load_weights(idx)
        for decoder_layer in self.layers:
            
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=None,
                    use_cache=False
                )

            hidden_states = layer_outputs[0]

        return hidden_states


class CustomedPhi3ForCausalLM(Phi3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi3
    def __init__(self, config):
        super().__init__(config)
        global device
        
        self.norm = Phi3RMSNorm(config.hidden_size).to(device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device)
        self.config = config

    def load_weights(self):
        global pre_weight_map, tensor_dict
        new_state_dict = {}
 
        new_state_dict['norm.weight'] = tensor_dict['model.norm.weight']
        new_state_dict['lm_head.weight'] = tensor_dict['lm_head.weight']
                    
        self.load_state_dict(new_state_dict)
        
        del tensor_dict['model.norm.weight']
        del tensor_dict['lm_head.weight']
    
    def load_one_file(self):
    
        global file_num
        
        if file_num > 6:
            print("파일 번호가 6번을 넘어감")
        file_path = self.config.base_path + f'/model-0000{file_num}-of-00006.safetensors'
        
        with safe_open(file_path, framework="pt", device="cuda") as f:
            for key in f.keys():
                tensor_dict[key] = f.get_tensor(key)
        
        file_num += 1
        
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        global file_num
        file_num = 1
        self.load_one_file()
        embed_model = EmbedModel(self.config)
        hidden_states = embed_model(input_ids)

        del embed_model

        past_seen_tokens = 0
        cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=device
            )
        position_ids = cache_position.unsqueeze(0)
        
        causal_mask = attention_mask

      
        body = Body(self.config.block_size, self.config)

        for idx in range(0, 40, self.config.block_size):
            outputs = body(idx, hidden_states, causal_mask, position_ids, None, cache_position)
            hidden_states = outputs

        del body

        hidden_states = outputs
        self.load_weights()
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        print('forward')

        return CausalLMOutputWithPast(
            logits=logits
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

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
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values =None,
        output_attentions= False,
    ):

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens =  0

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
                attention_mask.shape[-1]
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


        return causal_mask
    
    

    