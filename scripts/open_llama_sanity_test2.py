import textwrap
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

def print_response(response: str):
    """
    Print the response text with a specified line width.
    """
    print(textwrap.fill(response, width=110))

# Determine the device to use (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Special tokens and model configuration
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MAX_TOKENS = 1024
MODEL_NAME = 'openlm-research/open_llama_3b_v2'

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, add_eos_token=True, legacy=False)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME, local_files_only=False, torch_dtype=torch.float16, device_map="auto"
)
tokenizer.bos_token_id = BOS_TOKEN_ID

# Single prompt generation example
prompt = "The world's highest building is"
generation_config = GenerationConfig(max_new_tokens=256)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    tokens = model.generate(**inputs, generation_config=generation_config)

completion = tokenizer.decode(tokens[0], skip_special_tokens=True)

print("Question 1 - The world's highest building is \n")
print_response(completion)
print("\n")

# Function for top-k sampling
def top_k_sampling(logits, k=10):
    """
    Perform top-k sampling to select the next token.
    """
    top_k = torch.topk(logits, k)
    top_k_indices = top_k.indices
    top_k_values = top_k.values
    probabilities = torch.softmax(top_k_values, dim=-1)
    choice = torch.multinomial(probabilities, num_samples=1)
    token_id = int(top_k_indices[choice])
    return token_id

# Function to process chat with sampling
def process_chat(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, prompt: str, max_new_tokens: int = 256):
    """
    Generate a response to a prompt using the model with top-k sampling.
    """
    input_ids = tokenizer(prompt).input_ids
    output_token_ids = list(input_ids)

    max_src_len = MAX_TOKENS - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    with torch.inference_mode():
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=DEVICE),
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token_id]], device=DEVICE),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            token_id = top_k_sampling(last_token_logits)
            output_token_ids.append(token_id)

            if token_id == EOS_TOKEN_ID:
                break

    return tokenizer.decode(output_token_ids, skip_special_tokens=True)

# Generating responses for various prompts
prompts = [
    ("You're Michael G Scott from the office. What is your favorite phrase?", "Question 2 - You're Michael G Scott from the office. What is your favorite phrase? \n"),
    ("The world's highest building is", "Question 3 - The world's highest building is \n"),
    ("The best way to invest $10,000 is", "Question 4 - The best way to invest $10,000 is \n"),
    ("The best make and model v8 manual gearbox car is", "Question 5 - The best make and model v8 manual gearbox car is \n")
]

for prompt, question in prompts:
    response = process_chat(model, tokenizer, prompt)
    print(question)
    print_response(response)
    print("\n")
