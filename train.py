"""
Instruction fine-tuning of a huggingface transformer model using next-step prediction on everything.   

Usage:
    python train.py [--model MODEL] [--data DATA] [--lr LR] [--output OUTPUT] [--epochs EPOCHS] [--lora] [--wandb]

    Use python train.py -h to see the full list of arguments and their descriptions.
"""
from transformers import AutoModelForCausalLM, default_data_collator, TrainingArguments, AutoTokenizer
import argparse
import torch
from trl import SFTTrainer

DEFAULT_MODEL = "AI-Sweden-Models/gpt-sw3-126m"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

#TODO: Build a class with next-step prediction on the last bot response.
#Problem: Data is packed and does not keep track of indices. 

class CLMDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
        # Shift input_ids by one position to create the labels
        # Ensuring labels are shifted and the last position is ignored or masked
        self.labels = self.input_ids[:, 1:].contiguous()

    def __getitem__(self, idx):
        # For input_ids, take all tokens except the last one
        input_data = self.input_ids[idx, :-1]
        # For labels, all tokens are shifted by one, dropping the first token
        label_data = self.labels[idx]
        return {"input_ids": input_data, "labels": label_data}

    def __len__(self):
        return self.input_ids.size(0)
    

#TODO: NOT DONE!
def lora_train(model_id, train_data, eval_data, lr, output, wandb_log=False, epochs=3, batch_size=3):
    # Only import these if LoRA is used
    from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    
    if wandb_log:
        import wandb
        print("Select a name for the wandb run:")
        run_name = input()
        wandb.init(name=run_name)

    train_dataset = CLMDataset(train_data)
    eval_dataset = CLMDataset(eval_data)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, # Load model in 4bit mode
        bnb_4bit_use_double_quantization=True, # Nested quantization 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="causal_lm"
    )
    model = prepare_model_for_kbit_training(model)


    training_args = TrainingArguments(
        report_to="wandb" if wandb_log else None,   # enable logging to wandb
        output_dir=output,                      # output directory
        num_train_epochs=epochs,                # total number of training epochs
        per_device_train_batch_size=batch_size, # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=500,                       # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                      # strength of weight decay
        lr_scheduler_type='cosine',             # learning rate scheduler type
        learning_rate=lr,                       # learning rate
        logging_steps=10,                       # log every x updates
        evaluation_strategy="steps",            # evaluate every eval_steps
        eval_steps=100,                          # evaluation steps
        # gradient_accumulation_steps=2,        # gradient accumulation steps
        max_grad_norm=0.3,                      # max gradient norm
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=2048, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field='text',
        # compute_metrics=compute_metrics
    )

    model = get_peft_model(model, peft_config)
    print("Model loaded with LoRA.")
    trainer.train()
    # Save model to disk
    trainer.save_model()

    # train(model, train_data, eval_data, lr, output, wandb, epochs, batch_size=3)
def train(model, train_data, eval_data, lr, output, wandb_log=False, epochs=3, batch_size=3):
    if wandb_log:
        import wandb
        print("Select a name for the wandb run:")
        run_name = input()
        wandb.init(name=run_name)

    train_dataset = CLMDataset(train_data)
    eval_dataset = CLMDataset(eval_data)
    
    training_args = TrainingArguments(
        report_to="wandb" if wandb else None,   # enable logging to wandb
        output_dir=output,                      # output directory
        num_train_epochs=epochs,                # total number of training epochs
        per_device_train_batch_size=batch_size, # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=500,                       # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                      # strength of weight decay
        lr_scheduler_type='cosine',             # learning rate scheduler type
        learning_rate=lr,                       # learning rate
        logging_steps=10,                       # log every x updates
        evaluation_strategy="steps",            # evaluate every eval_steps
        eval_steps=50,                          # evaluation steps
        # gradient_accumulation_steps=2,        # gradient accumulation steps
        max_grad_norm=0.3,                      # max gradient norm
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=2048, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field='text',
        # compute_metrics=compute_metrics
    )
    print(f"Training model with learning rate {lr}, output directory {output} and wandb logging set to {wandb_log}.")
    trainer.train()
    # Save model to disk
    trainer.save_model()
    print("Model saved to disk.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="The name of (or path to) the HuggingFace transformer model to use for training. Default is 'AI-Sweden-Models/gpt-sw3-126m'.")
    parser.add_argument('--data', type=str, help="The name of train and eval data to use for training, without adding '_train.pt' or '_eval.pt' as this is already assumed.")
    parser.add_argument('--lr', type=float, default=1e-4, help="The learning rate to use for training. Default is 1e-4.")
    parser.add_argument('--output', type=str, default="./results", help="The directory to save the trained model to. Default is './results'.")
    parser.add_argument('--epochs', type=int, default=3, help="The number of epochs to train for. Default is 3.")
    parser.add_argument('--lora', action='store_true', help="Whether to use LoRA for training. Default is False.")
    parser.add_argument('--wandb', action='store_true', help="Whether to use wandb for logging. Default is False.")
    parser.add_argument('--batch_size', type=int, default=3, help="The batch size to use for training. Default is 3.")
    parser.set_defaults(lora=False, wandb=False)

    args = parser.parse_args()

    if args.data:
        train_data = torch.load(args.data + '_train.pt')
        eval_data = torch.load(args.data + '_eval.pt')
    else:
        print("No data provided. Exiting.")
        exit(1)
        
    if args.model:
        if args.lora:
            lora_train(args.model, train_data, eval_data, args.lr, args.output, args.wandb, args.epochs, args.batch_size)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
            train(model, train_data, eval_data, args.lr, args.output, args.wandb, args.epochs, args.batch_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, torch_dtype=torch.bfloat16).to(device)
        train(model, train_data, eval_data, args.lr, args.output, args.wandb, args.epochs, args.batch_size)

