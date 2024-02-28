"""
Datahandler tokenizes raw data into instruction-tune format.
May also upload the tokenized tensors to HuggingFace Model Hub as a dataset.

Usage:
    If used natively. Run the following command in the terminal:

    python datahandler.py --model [model_name] --file [filename]
    
    model_name: The name of the HuggingFace model to use for tokenization. Default is "AI-Sweden-Models/gpt-sw3-126m"
    filename: The name of the jsonl file to tokenize.
"""
from transformers import AutoTokenizer
import torch 
import argparse
import json 

ROLEMAP = {'<human>': 'User', 'content': 'User', '<bot>': 'Bot'}
default_model = "AI-Sweden-Models/gpt-sw3-126m"
tokenizer = AutoTokenizer.from_pretrained(default_model)

def handle_data(file, upload_to_hub=False):
    bos_token = tokenizer.special_tokens_map['bos_token']
    eos_token = tokenizer.special_tokens_map['eos_token']

    return_tensor = torch.tensor([])
    with open(file, 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            return_tensor = torch.cat((return_tensor, tokenize(line, bos_token, eos_token)), 0)

    print(return_tensor.shape)

    if upload_to_hub:
        print("Uploading to HuggingFace as dataset")
        # Upload to HuggingFace Model Hub
        pass


def tokenize(line : dict, bos_token : str, eos_token : str) -> torch.Tensor:
    """
    Tokenizes a single line of data into instruction format, in the manner given in the example below, and stores it in a tensor.
    
    Example of format:\n
    <|endoftext|>\n
    <s> User\n
    Hello, how are you?\n
    <s> Bot\n
    I am fine, thank you.\n
    <s> ... 
    """
    turns = line['text']
    output = [
        f"{bos_token} {ROLEMAP[role]}\n{msg}\n"
        for turn in turns
        for role, msg in turn.items()
    ]
    output.append(bos_token)
    output = eos_token + '\n' + ''.join(output)

    output_tensor = tokenizer.encode(output, return_tensors="pt", padding=False)
    return output_tensor

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--file', type=str, default=None)    

    args = parser.parse_args()
    if args.model != default_model and args.model: 
        try:
            model = args.model
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except:
            print("Model not found, please check model name. E.g. AI-Sweden-Models/gpt-sw3-126m")
            exit()
    else:
        print("No model specified for tokenization, using default model: AI-Sweden-Models/gpt-sw3-126m")

    if args.file:
        handle_data(args.file)
    else:
        print("No file specified, please specify a file to tokenize using flag '--file [filename]'")
        exit()
