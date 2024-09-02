from datasets import load_dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline_meta import CustomedPipeline
from hf import NewPhi3Config
from model_meta import CustomedPhi3ForCausalLM


torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("allenai/openbookqa", split='validation')
test = dataset.select(range(5))
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

#model_inputs = dataset.map(preprocess, batched=True, batch_size=5)
model_inputs = test.map(preprocess, batched=True, batch_size=5)

pipe = CustomedPipeline(model, config)
inputs = pipe.preprocess(model_inputs)
print("hi")
outputs = pipe.forward(inputs)
result = pipe.postprocess(outputs, inputs['labels'])
print(result)
