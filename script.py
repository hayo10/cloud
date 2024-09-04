from datasets import load_dataset
import torch
from transformers import AutoTokenizer
import torch
import gc
from customed_pipeline import CustomedPipeline
from hf import NewPhi3Config
from safetensors import safe_open
from model import CustomedPhi3ForCausalLM
import gc
import os
import time

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("allenai/openbookqa", split='validation')
# test = dataset.select(range(20))
config = NewPhi3Config()
model = CustomedPhi3ForCausalLM(config)
pipe = CustomedPipeline(model, config)

prefix = "\nRead the question and answer the following sentence in given multiple choice.\nAnswer only the sentence you chose. Never include a question and other word in your answer.\n\nquestion: "

def preprocess(data):
    model_inputs = []
    for i in range(len(data['question_stem'])):
        offset = ord(data['answerKey'][i]) - ord('A')
        chat_dict = {
            "messages" :[
                {
                    "role" : "user",
                    "content" : prefix + data['question_stem'][i] + "\nchoices: ["
                }
            ],
            "answer" : data['choices'][i]['text'][offset]
        }
        
        for j in range(4):
            chat_dict['messages'][0]['content'] += "\'" + data['choices'][i]['text'][j]
            if j < 3:
                chat_dict['messages'][0]['content'] += "\', "
            else:
                chat_dict['messages'][0]['content'] += "\']\n"
        
        model_inputs.append(chat_dict)
    
    # return the processed data as a dict
    return {"processed_data": model_inputs}

def batchify(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

#model_inputs = dataset.map(preprocess, batched=True, batch_size=5)
model_inputs = dataset.map(preprocess, batched=True, batch_size=20)
model_inputs = model_inputs['processed_data']
messages = [inputs['messages'] for inputs in model_inputs]
labels = [inputs['answer'] for inputs in model_inputs]

batch_size = 5
batches = batchify(messages, batch_size)
tokenized = [tokenizer.apply_chat_template(batch, 
                                          tokenize=True, 
                                          padding=True, 
                                          truncation=True,
                                          return_tensors="pt", 
                                          return_dict=True) for batch in batches]
input_ids = [token['input_ids'] for token in tokenized]
attention_mask = [token['attention_mask'] for token in tokenized]


pipe = CustomedPipeline(model, config)
outputs = pipe.forward(input_ids, attention_mask)
result = pipe.postprocess(outputs, labels)
print(result)

# for batch in zip(input_ids, attention_mask):
#     sentence = generate(batch[0], batch[1], 1, 32000)
#     print(sentence)

