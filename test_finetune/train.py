"""
Instruction fine-tuning of a model. 


"""
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import argparse
import torch


DEFAULT_MODEL = "AI-Sweden-Models/gpt-sw3-126m"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 2

def prepare_for_training(model, data):
    train_dataloader = DataLoader(data, 
                                  batch_size=batch_size,
                                  shuffle=False)


def train(model, data):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()

    if args.data:
        data = torch.load(args.data)
    else:
        print("No data provided. Exiting.")
        exit(1)
        
    if args.model: 
        model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL).to(DEVICE)


    
     

