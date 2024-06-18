import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Path to the model
model_path = 'openlm-research/open_llama_3b_v2'

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)

# Load the model with specified dtype and device mapping
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto'
)

# Define the prompt
prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate the response
generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)

# Decode and print the generated response
print(tokenizer.decode(generation_output[0]))
