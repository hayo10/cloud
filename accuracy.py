

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc
from torch.utils.data import DataLoader

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("allenai/openbookqa", split='validation')

print(dataset)

test = dataset.select(range(20))


for item in dataset:
    # Use a default value or empty string if 'fact1' is missing
    print(item)

max_length = 100
prefix = "Which of A, B, C, or D is the most possible word to follow the sentence? sentece:"
model_inputs = {
    'input_ids': [],
    'attention_mask': [],
    'labels': []
}
suffix = "Choose only one from A,B,C and D. answer:"

def preprocess(data):
    inputs = []
    labels = []
    print(data)
    for i in range(len(data['question_stem'])):
        quiz = prefix + data['question_stem'][i][:-1]

        for j in range(4):
            quiz += data['choices'][i]['label'][j] + ":" + data['choices'][i]['text'][j]  + suffix

        inputs.append(quiz)
        labels.append(data['answerKey'][i])


    tokenized_inputs= tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    tokenized_labels = tokenizer(labels, return_tensors="pt")

    model_inputs['input_ids'].append(tokenized_inputs['input_ids'])
    model_inputs['attention_mask'].append(tokenized_inputs['attention_mask'])
    model_inputs['labels'].append(tokenized_labels['input_ids'])

tokenized = test.map(preprocess, batched=True, batch_size=2)

model_inputs['input_ids'] = torch.vstack(model_inputs['input_ids'])
model_inputs['attention_mask'] = torch.vstack(model_inputs['attention_mask'])
model_inputs['labels'] = torch.cat(model_inputs['labels'], dim=0)

model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}

decoded_labels = [tokenizer.decode(label_ids, skip_special_tokens=True) for label_ids in model_inputs['labels']]


allowed_tokens = ['A', 'B', 'C', 'D']
allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)

with torch.no_grad():
    outputs = model(input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'])

processed = torch.nn.functional.softmax(outputs.logits[:,-1,:], dim=1)

generated = torch.argmax(processed,dim=1)

decoded_outputs = [tokenizer.decode(answer, skip_special_tokens=True) for answer in generated]
print(decoded_outputs)



with torch.no_grad():
    outputs = model(input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask']
                            )


suffix = "answer:"
print(outputs)
generated_output = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
position = [output.find(suffix) for output in generated_output]

print(generated_output[max_length:])

class CustomedPipeline():
    def __init__(
            self,
            config,
            model_id = "microsoft/Phi-3-mini-4k-instruct",
            device = "cuda"
        ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model =  CustomedPhi3ForCausalLM(self.tokenizer, self.config)


    def forward(self, model_inputs, max_length = 500):
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        prompt_len = model_inputs['prompts']

        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=max_length)
        return {"generated_sequence": generated_sequence, "prompt_len" :prompt_len}

    def postprocess(self, model_outputs, clean_up_tokenization_spaces=True):
        generated_sequence = model_outputs["generated_sequence"]
        prompt_len = model_outputs["prompt_len"]

        result = []

        for i, text in enumerate(generated_sequence):
            eos_pos = (text == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_pos) > 0:
                eos_after_prompt = next((pos.item() for pos in eos_pos if pos.item() > prompt_len), None)

                if eos_after_prompt is not None:
                    text = text[prompt_len:eos_after_prompt-1]
                else:
                    text = text[prompt_len:]
            else:
                text = text[prompt_len:]

            #decoded_text = self.tokenizer.decode(text, skip_special_tokens=True)
            decoded_text = self.tokenizer.decode(text)
            result.append([{'generated':decoded_text}])

        return result

gc.collect()