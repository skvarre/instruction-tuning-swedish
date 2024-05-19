"""
This updated version of train.py performs finetuning by letting
the SFTTrainer class handle the preprocessing part, without relying on datahandler.py.     
"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import torch 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from datasets import load_dataset
from math import sqrt 

MAX_SEQ_LENGTH = 2048 # Max-seq length for gpt-sw3 models
BF16_SUPPORT = torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16_SUPPORT else torch.float16

"""Hyperparameters for fine-tuning"""
gradient_accumulation_steps = 25
learning_rate = 2e-4
batch_size = 8
epochs = 3
lr_scheduler_type = "cosine"
warmup_steps = 500
weight_decay = 0.01
optimizer = "adamw_8bit"
use_gradient_checkpointing = True
"""LoRA hyperparameters"""
q_lora = True # Set to false for full finetune without quantization
lora_alpha = 16
lora_dropout = 0
lora_rank = 64


"""Format the prompts, assumes standard conversational turns"""
def formatting_prompts(examples):
    convs = examples['conversations']
    texts = []
    mapper = { "system": "<|endoftext|><s>\nSYSTEM\n", "human": "USER:\n", "gpt" : "ASSISTANT:\n"}
    end_mapper = {"system" : "\n\n<s>", "human": "\n<s>", "gpt": "\n<s>"}
    for convo in convs:
        text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}{end_mapper[turn]}" for x in convo)
        texts.append(text)
    return {"text": texts}

"""Format prompts for translation task."""
def formatting_translation(examples):
    prompt_pairs = [
        f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{en}<s>Bot: {sv}<s>"
        for en, sv in zip(examples["en"], examples["sv"])
    ]
    return {"text": prompt_pairs}

def train(model_id, dataset, output, split, wandb_log=False):

    dataset = dataset['train'].train_test_split(test_size=split)
    dataset = dataset.map(formatting_prompts, batched=True,)
    steps_per_epoch = len(dataset['train']) // (batch_size * gradient_accumulation_steps)
    
    if wandb_log:
        print("Select a name for the wandb run:")
        run_name = input()
        print("Select project to store run in. Leave blank for default.")
        project_name = input()
        wandb.init(name=run_name, project=project_name if project_name != "" else None)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quantization=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        torch_dtype = DTYPE,
        quantization_config = bnb_config if q_lora else None,
        token = None,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=MAX_SEQ_LENGTH
    )

    lora_config = LoraConfig(
        lora_alpha=lora_alpha,       
        lora_dropout=lora_dropout,
        r=lora_rank,               
        bias="none",
        task_type="causal_lm",
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        report_to="wandb" if wandb_log else None,
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
        eval_steps=steps_per_epoch//4,
        evaluation_strategy="steps",
        do_eval=True,
        do_train=True,
        max_grad_norm=1.0 * sqrt(gradient_accumulation_steps),
        optim=optimizer,
        fp16 = not BF16_SUPPORT,
        bf16 = BF16_SUPPORT,
        logging_steps=1,
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        save_steps=steps_per_epoch,
        save_total_limit=3,        
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field='text',
        packing=False
    )

    trainer.train() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with SFT.')
    parser.add_argument('--model', type=str, default="AI-Sweden-Models/gpt-sw3-126m", help='Model to use for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--output', type=str, default="./models", help='Output directory for model.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size.')
    parser.add_argument('--dataset', type=str, help='Dataset to use.')
    parser.add_argument('--wandb', action='store_true', help="Whether to use wandb for logging. Default is False.")
    parser.add_argument('--split', type=float, default=0.2, help='Train-test split ratio.')
    parser.add_argument('--hf', action='store_true', help="Whether to use HuggingFace dataset.")
    parser.set_defaults(wandb=False)
    parser.set_defaults(hf=False)
    args = parser.parse_args()

    if args.dataset:
        try: 
            if args.hf:
                dataset = load_dataset(args.dataset)
            else:
                # Running local dataset. 
                dataset = load_dataset('json', data_files=args.dataset)   
                dataset = dataset.shuffle()             
        except:
            print("Dataset not found, please check dataset name.")
            exit()
    else: 
        print("Dataset not found, please check dataset name.")
        exit()

    learning_rate = args.lr
    output = args.output
    epochs = args.epochs
    batch_size = args.batch_size
        
    if args.model:
        model = args.model
    else:
        print("Model not found, please check model name. E.g. AI-Sweden-Models/gpt-sw3-126m")
        exit()

    train(model, dataset, output, args.split, args.wandb)

