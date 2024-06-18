import argparse
import os
from datasets import load_dataset
import torch
import transformers
from accelerate import PartialState
from peft import LoraConfig
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer

def get_args():
    """
    Parse and return command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--dataset_name", type=str, default="the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/rust")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="content")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )

def main(args):
    # Configuration for quantization and LoRA (Low-Rank Adaptation)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",
    )

    # Load model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
    model = LlamaForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )

    #set configs
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("Model parameters...")
    print_trainable_parameters(model)

    # Load dataset
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/ada", split="train")

    # Determine the number of training samples and epochs
    num_train_samples = len(ds)
    print(f'Number of training samples: {num_train_samples}')
    epochs = args.max_steps / num_train_samples
    print(f'Number of training epochs: {epochs}')

    # Setup the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=5,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to="wandb",
        ),
        peft_config=lora_config,
        dataset_text_field=args.dataset_text_field,
        packing=False,
    )

    # Launch training
    print("Training...")
    trainer.train()

    # Save the final checkpoint of the model and tokenizer
    output_dir = os.path.join(args.output_dir, "final_checkpoint/")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir, safe_serialization=False)
    trainer.tokenizer.save_pretrained(output_dir)

    print("Training Done! 💥")

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.set_verbosity_error()
    main(args)
