from transformers import AutoTokenizer
from customed_pipeline import CustomedPipeline
from hf_ref import NewPhi3Config
from model2 import CustomedPhi3ForCausalLM

model_id = "microsoft/Phi-3-medium-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

base_path = 
config = NewPhi3Config(base_path=base_path)
model = CustomedPhi3ForCausalLM(config)

pipe = CustomedPipeline(model, config)
pipe.load_data(dataset_name = "allenai/openbookqa", split='validation', batch_size=40)
outputs = pipe.forward(max_new_tokens=15)
result = pipe.postprocess(outputs)


            