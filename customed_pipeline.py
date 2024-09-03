# -*- coding: utf-8 -*-
"""customedPipeline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11o3lLBWeIGO4_yRU4NIre7XC1ps4xaDM
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import Phi3Config

import gc
gc.collect()

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="cuda",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )


class CustomedPipeline():
    def __init__(
            self,
            model,
            config,
            model_id = "microsoft/Phi-3-medium-4k-instruct",
            device = "cuda"
        ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model =  model

    # def preprocess(
    #         self,
    #         prompt_text,
    #         prefix="",
    #         handle_long_generation=None,
    #         add_special_tokens=None
    #         ):

    #     inputs = self.tokenizer.apply_chat_template(
    #             prompt_text,
    #             add_generation_prompt=True,
    #             tokenize=True,
    #             return_tensors="pt",
    #         return_dict=True,
    #             padding=True
    #         ).to('cuda')
    #     inputs['prompts'] = inputs['input_ids'].shape[-1]

    #     return inputs

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids[0].to('cuda')
        attention_mask = attention_mask[0].to('cuda')
        prompt_len = input_ids.shape[1]

        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=prompt_len+15)
        return {"generated_sequence": generated_sequence, "prompt_len" :prompt_len}

    def find_pattern(self, text):
        cnt = 0
        idx = []
        for i in range(len(text)-1,0,-1):
            if text[i] ==32001 and text[i-1] == 32007:
                idx.append(i)
                if len(idx) == 2:
                    break
        if len(idx) == 2:
            result = text[idx[1]+1:idx[0]-1]
        else:
            result = text[idx[0]+1:]
        return result

    def postprocess(self,model_outputs, labels, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"]

        result = []

        correct = 0

        for i, text in enumerate(generated_sequence):
            answer = self.find_pattern(text)
            decoded_answer = self.tokenizer.decode(answer)
            if decoded_answer == labels[i]:
                correct += 1
            result.append([{'generated':decoded_answer, 'label' : labels[i]}])
            

        print('accuracy : ',correct/len(generated_sequence))
        return result


