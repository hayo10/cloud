import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import json

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
        self.input_ids = []
        self.attention_mask = []
        self.device = device
        self.labels = []

    def batchify(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    
    def load_data(self, file_path, batch_size):
        model_inputs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                model_inputs.append(json_obj)
        messages = [inputs['message'] for inputs in model_inputs]
        self.labels = [inputs['answer'] for inputs in model_inputs]
        
        batches = self.batchify(messages, batch_size)
        tokenized = [self.tokenizer.apply_chat_template(batch, 
                                                tokenize=True, 
                                                padding=True, 
                                                truncation=True,
                                                return_tensors="pt", 
                                                return_dict=True) for batch in batches]
        
        self.input_ids = [token['input_ids'] for token in tokenized]
        self.attention_mask = [token['attention_mask'] for token in tokenized]
            
    def forward(self, max_new_tokens):
        times = 0
        result = []
        
        for batch in zip(self.input_ids, self.attention_mask):
            st = time.time()
            inputs = batch[0].to(self.device)
            masks = batch[1].to(self.device)
            generated_sequence = self.model.generate(input_ids=inputs, attention_mask=masks, max_new_tokens=max_new_tokens )
            end = time.time()
            result.append(generated_sequence)
            print('batch load and inference time ', end - st)
            times += end - st
        return {"generated_sequence": result}

    def find_pattern(self, text):
        idx = []
        for i in range(91,len(text)-1):
            if text[i] == 32007 and text[i+1]==32001:
                idx.append(i)
           
        if len(idx) == 0:
            return text[90:]
        elif len(idx) == 1:
            return text[idx[0]+2:]
        else:
            return text[idx[0]+2: idx[1]]

    def postprocess(self,model_outputs,  clean_up_tokenization_spaces=True):

        result = []
        correct = 0
        
        for outputs in model_outputs['generated_sequence']:
            for i, text in enumerate(outputs):
                answer = self.find_pattern(text)
                decoded_answer = self.tokenizer.decode(answer)
                if self.labels[i] in decoded_answer:
                    correct += 1
                else:
                    decoded_answer = self.tokenizer.decode(text[91:])
                result.append([{'generated':decoded_answer, 'label' : self.labels[i]}])

        total = len(self.labels)
        print('맞은 개수', correct)
        print('총 개수 ',total)

        print('accuracy : ', float(correct/total))
        return result


