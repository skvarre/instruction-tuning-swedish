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
from peft import AutoPeftModelForCausalLM

DEFAULT_MODEL = "AI-Sweden-Models/gpt-sw3-126m"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 2
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

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
    

# I would do this, except the packing of the data prevents it
# since the last token is not always the last token in the tensor.
# TODO: Fix datahandler to allow for this.
# class CLMDataSetLastToken(torch.utils.data.Dataset):

def test():
    pass
        
def train(model, train_data, eval_data, lr, output, wandb=False, epochs=3):

    train_dataset = CLMDataset(train_data)
    eval_dataset = CLMDataset(eval_data)
    
    training_args = TrainingArguments(
        report_to="wandb" if wandb else None, # enable logging to wandb
        output_dir=output,                    # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=3,        # batch size per device during training
        per_device_eval_batch_size=3,         # batch size for evaluation
        warmup_steps=500,                     # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                    # strength of weight decay
        lr_scheduler_type='cosine',           # learning rate scheduler type
        learning_rate=lr,                     # learning rate
        logging_steps=10,                     # log every x updates
        evaluation_strategy="steps",          # evaluate every eval_steps
        eval_steps=20,                        # evaluation steps
        # gradient_accumulation_steps=2,        # gradient accumulation steps
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=2048,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field='text',
    )
    print(f"Training model with learning rate {lr}, output directory {output} and wandb logging set to {wandb}.")
    trainer.train()
    # Save model to disk
    trainer.save_model()
    print("Model saved to disk.")

#TODO: Implementation of wandb logging is not correct. type=bool behaves differently than intuitive.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="The name of (or path to) the HuggingFace transformer model to use for training. Default is 'AI-Sweden-Models/gpt-sw3-126m'.")
    parser.add_argument('--data', type=str, help="The name of train and eval data to use for training, without adding '_train.pt' or '_eval.pt' as this is already assumed.")
    parser.add_argument('--lr', type=float, default=1e-4, help="The learning rate to use for training. Default is 1e-4.")
    parser.add_argument('--output', type=str, default="./results", help="The directory to save the trained model to. Default is './results'.")
    parser.add_argument('--epochs', type=int, default=3, help="The number of epochs to train for. Default is 3.")
    parser.add_argument('--lora', action='store_true', help="Whether to use LoRA for training. Default is False.")
    parser.add_argument('--wandb', action='store_true', help="Whether to use wandb for logging. Default is False.")
    parser.set_defaults(lora=False, wandb=False)

    args = parser.parse_args()

    if args.data:
        train_data = torch.load(args.data + '_train.pt')
        eval_data = torch.load(args.data + '_eval.pt')
    else:
        print("No data provided. Exiting.")
        exit(1)
        
    if args.model: 
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL).to(device)


    # train(model, train_data, eval_data, args.lr, args.output, args.wandb, args.epochs)