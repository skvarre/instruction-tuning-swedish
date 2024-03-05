"""
Perform inference on a huggingface transformer model in an instruction manner, using
AutoModelForCausalLM and AutoTokenizer.

Usage:
    python inference.py [--model MODEL]
    model: 
        The name of the HuggingFace transformer model to use for inference. Default is "AI-Sweden-Models/gpt-sw3-126m" 
    parse:
        Whether to parse the input prompt into a format that the model can understand. Default is True.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#TODO: Hardcoded. Assumes bos_token = <s> and eos_token = <|endoftext|>.
def parse_input(prompt, beginning_of_conversation):
    parsed_output = f"\n<s>User\n{prompt}\n<s>Bot"
    return f"<|endoftext|>{parsed_output}" if beginning_of_conversation else parsed_output
    # return f"<|endoftext|>\n<s>User\n{prompt}\n<s>Bot"

def generate(model, tokenizer, prompt, max_length=200):    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Stop at the first <s> token.
    return "".join(output.split('<s>')[0:3])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="results")
    parser.add_argument('--p', type=str, default='True')
    args = parser.parse_args()
    model_path = args.model if args.model else "results"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded on {"GPU" if device == "cuda:0" else "CPU"}.")

    beginning_of_conversation = True # Not sure this is needed. 
    while True:
        print("Type prompt or press ENTER to exit:")
        prompt = input()

        if prompt == "":
            print("Exiting...")
            exit(0)
        else:
            parsed_prompt = parse_input(prompt, beginning_of_conversation) if args.p == 'True' else prompt
            print(generate(model, tokenizer, parsed_prompt))
            beginning_of_conversation = False