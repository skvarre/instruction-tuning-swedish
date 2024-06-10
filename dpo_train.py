"""
Trainer script for DPO alignment.
"""
import argparse
from trl import DPOTrainer 
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
import torch 
from datasets import load_dataset
from peft import LoraConfig
import wandb 

USE_SYSTEM_PROMPTS = False # Append system prompts to question.
MAX_SEQ_LENGTH = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
gradient_accumulation_steps = 10

def return_prompt_and_responses(samples, use_system=False) -> dict[str, str, str]:
    return {
        "prompt": f"<|endoftext|><s>\nUSER:{samples["system"]}\n\n{samples["question"]}\n<s>" if samples["system"] != "" else f"<|endoftext|><s>\nUSER:\n{samples["question"]}\n<s>",
        "chosen": f"ASSISTANT:\n{samples["chosen"]}\n<s>",
        "rejected": f"ASSISTANT:\n{samples["rejected"]}\n<s>"
    }



def train(args):
    wandb_log = args.wandb
    if wandb_log:
        print("Select a name for the wandb run:")
        run_name = input()
        print("Select project to store run in. Leave blank for default.")
        project_name = input()
        wandb.init(name=run_name, project=project_name if project_name != "" else None)

    effective_batch_size = 60
    gradient_accumulation_steps = effective_batch_size // args.batch_size

    model_path = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )

    peft_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
    )
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        quantization_config=bnb_config
    )
    print("Model loaded.")

    dataset = load_dataset("json", data_files=args.dataset)
    dataset = dataset["train"]
    original_columns = dataset.column_names

    dataset = dataset.map(
        return_prompt_and_responses,
        # batched=True,
        remove_columns=original_columns
    )
    # Create train and test split
    dataset = dataset.train_test_split(test_size=args.split)
    steps_per_epoch = len(dataset['train']) // (args.batch_size * gradient_accumulation_steps)

    training_args = TrainingArguments(
        report_to="wandb" if wandb_log else None,
        output_dir=args.output,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        eval_steps=steps_per_epoch//4,
        evaluation_strategy="steps",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_grad_norm=0.3,
        save_steps=steps_per_epoch,
        save_total_limit=3,   
        metric_for_best_model="eval_loss",
        label_names=["labels"],
        do_eval=True,        
    )

    dpo_trainer = DPOTrainer(
        model=model,
        peft_config=peft_config,
        beta=0.1,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_SEQ_LENGTH / 2,
        args=training_args,
    )

    dpo_trainer.train()
    dpo_trainer.save_model()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with SFT.')
    parser.add_argument('--model', type=str, default="AI-Sweden-Models/gpt-sw3-126m", help='Model to use for training.')
    parser.add_argument('--base-model', type=str, help='Freezed reference model. If not provided, the model will be used as the base model.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--output', type=str, default="./models", help='Output directory for model.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size.')
    parser.add_argument('--dataset', type=str, help='Dataset to use.')
    parser.add_argument('--wandb', action='store_true', help="Whether to use wandb for logging. Default is False.")
    parser.add_argument('--split', type=float, default=0.15, help='Train-test split ratio.')
    parser.add_argument('--hf', action='store_true', help="Whether to use HuggingFace dataset.")
    parser.set_defaults(wandb=False)
    parser.set_defaults(hf=False)
    args = parser.parse_args()

    train(args)


