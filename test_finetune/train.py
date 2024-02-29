"""
Instruction fine-tuning of a model. 

Usage:
    python train.py [--model MODEL] [--data DATA]

    model: 
        The name of the HuggingFace transformer model to use for training. Default is "AI-Sweden-Models/gpt-sw3-126m"
    data: 
        The name of train and eval data to use for training, without "_train.pt" or "_eval.pt" added as this is already assumed.
    """
from transformers import AutoModelForCausalLM, default_data_collator
from torch.utils.data import DataLoader
import argparse
import torch


DEFAULT_MODEL = "AI-Sweden-Models/gpt-sw3-126m"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 2

def prepare_for_training(model, train_data, eval_data):

    train_dataloader = DataLoader(train_data, 
                                  batch_size=batch_size,
                                  shuffle=False)
    
    eval_dataloader = DataLoader(eval_data,
                                 batch_size=batch_size,
                                 shuffle=False)
    

    

def train(model, data):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--data', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
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

    prepare_for_training(model, train_data, eval_data)


    
     

