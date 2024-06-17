"""
Script to Calculate the perplexity on a given model using a given dataset.
"""
import argparse
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate(model, tokenizer, dataset, max_length, batch_size):
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, remove_columns=dataset["train"].column_names)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= tokenizer.model_max_length:
            total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
        result = {k: [t[i:i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
                  for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_datasets = tokenized_dataset.map(group_texts, batched=True)
    dataloader = DataLoader(lm_datasets["train"], batch_size=batch_size, shuffle=True)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in dataloader:
        inputs, labels = batch["input_ids"].to(DEVICE), batch["labels"].to(DEVICE)

        with torch.inference_mode():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * inputs.size(1)
            total_tokens += inputs.size(1)
    
    perplexity = torch.exp(total_loss / total_tokens)
    return perplexity.item()

    
def calculate_perplexity(args):
    output = args.output + args.model.split("/")[-1] + "-" + args.dataset.split("/")[-1] + ".txt"

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=args.max_length)
    except:
        print(f"Error loading model from {args.model}.")
        return
    
    try:
        if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
            dataset = load_dataset("json", data_files=args.dataset)
        elif args.dataset.endswith(".csv"):
            dataset = load_dataset("csv", data_files=args.dataset)
        else:
            dataset = load_dataset(args.dataset)
    except:
        print(f"Error loading dataset from {args.dataset}.")
        return
    
    perplexity = calculate(model, tokenizer, dataset, args.max_length, args.batch_size)
    print(f"Perplexity: {perplexity}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Perplexity on a Model.')
    parser.add_argument('--model', type=str, help='Local/HF Path to the model to use.')
    parser.add_argument('--dataset', type=str, help='Dataset to run perplexity on.')
    parser.add_argument('--max_length', type=int, default=2048, help='Max model sequence length.')
    parser.add_argument('--output', type=str, default="./results/perplexity-scores/", help='Output directory for results.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    args = parser.parse_args()

    calculate_perplexity(args)        
