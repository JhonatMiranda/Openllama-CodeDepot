import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

# Define the path to the base and fine-tuned models
base_model_path = 'open_llama_3b_v2'
finetuned_model_path = 'open_llama_3b_finetuned/final_checkpoint'

# Load the tokenizer from the base model path
tokenizer = LlamaTokenizer.from_pretrained(base_model_path, legacy=False, use_fast=False)

# Load the base model with specified dtype and device mapping
model = LlamaForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map='auto')

# Load the fine-tuned model
finetuned_model = PeftModel.from_pretrained(model, finetuned_model_path)

# Merge the base and fine-tuned models
model = finetuned_model.merge_and_unload()

# Configure tokenizer settings
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set the model to evaluation mode
model.eval()

# Function to generate and print responses for given prompts
def generate_and_print(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=128)
    response = tokenizer.decode(generation_output[0])
    print(response)
    print("\n")

# List of test prompts
prompts = [
    "What is the largest animal?\n",
    "The world's highest building is?\n",
    "You're Michael G Scott from the office. What is your favorite phrase?\n",
    "The best way to invest $10,000 is?\n",
    "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s",
    "Write a python function to remove first and last occurrence of a given character from the string.\n",
    "You are an expert programmer that helps to write JavaScript code based on the user request, with concise explanations. Don't be too verbose.\n",
    "Table departments, columns = [DepartmentId, DepartmentName] \n Table students, columns = [DepartmentId, StudentId, StudentName] \n Create a MySQL query for all students in the Computer Science Department\n",
    "django view for rendering the current day and time without a template \n  def current_datetime(request):\n",
    "A simple python function to remove whitespace from a string:\n",
    "modes = ['r', 'w', 'a', 'r+']\n\ndef validate_file_name(file_name):\n    if file_name > '':\n        return True\n    return False\n",
    "N = input()\n\nheights_str = input()\nstalls_str= input()\n\nN = int(N)\n\nheights = heights_str.split()\nheights = [int(h) for h in heights]\n\nheights.sort(reverse=True)\n\nstalls = stalls_str.split()\nstalls = [int(s) for s in stalls]\n\nstalls.sort(reverse=True)\n\ncnt = 0\n\ndef cal_cnts(heights, stalls):\n",
    "import pathlib\n\nfrom typing import List\n\nimport numpy\n\ndef read_inputs(input_file: str) -> List[int]:\n",
    "with Arduino_Nano_33_Ble_Sense.IOs;\nwith Ada.Real_Time; use Ada.Real_Time;\n\nprocedure Main is\n   TimeNow : Ada.Real_Time.Time := Ada.Real_Time.Clock;\nbegin\n",
    "with Text_Io; use Text_Io;\nwith Revisions; use Revisions;\n\nprocedure Test is\nbegin\n",
    "TimeNow := Ada.Real_Time.Clock;\n",
    "Write code in Ada using Revisions and Text_IO to perform a test procedure with branch and commit Put_Line's\n"
]

# Generate and print responses for each prompt
for i, prompt in enumerate(prompts):
    print(f"Test {i + 1} --------------------------------")
    generate_and_print(prompt)
