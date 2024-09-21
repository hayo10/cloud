from datasets import load_dataset
import torch
from transformers import AutoTokenizer
import torch
from customed_pipeline import CustomedPipeline
from hf_ref import NewPhi3Config
from model2 import CustomedPhi3ForCausalLM


torch.random.manual_seed(0)

model_id = "microsoft/Phi-3-medium-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = NewPhi3Config()
model = CustomedPhi3ForCausalLM(config)

pipe = CustomedPipeline(model, config)
pipe.load_data(dataset_name = "allenai/openbookqa", split='validation', batch_size=40)
outputs = pipe.forward(max_new_tokens=15)
result = pipe.postprocess(outputs)
print(result)

 