"""
Perform inference on a huggingface transformer model in an instruction manner, using
AutoModelForCausalLM and AutoTokenizer.

Usage:
    python inference.py [--model MODEL]
    model: 
        The name of (or path to) the HuggingFace transformer model to use for inference.
    parse:
        Whether to parse the input prompt into a format that the model can understand. Default is True.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import argparse
import torch 

device = "cuda:0" if torch.cuda.is_available() else "cpu"


#TODO: Hardcoded. Assumes bos_token = <s> and eos_token = <|endoftext|>.
def parse_input(prompt):
    # parsed_output = f"\n<s>User\n{prompt}\n<s>Bot\n"
    # return f"<|endoftext|>{parsed_output}" if beginning_of_conversation else parsed_output
    return f"<|endoftext|>\n<s>User\n{prompt}\n<s>Bot\n"

def generate(model, tokenizer, prompt, max_length=200):    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Stop at the first <s> token.
    return "".join(output.split('<s>')[0:3])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/results")
    parser.add_argument('--lora', action='store_true', help="Whether to use LoRA for inference. This assumes adapters as model argument. Default is False.")
    parser.set_defaults(lora=False)

    args = parser.parse_args()

    if args.lora:

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                     # Load model in 4-bit mode
            bnb_4bit_use_double_quantization=True, # Nested quantization 
            bnb_4bit_quant_type="nf4",             # Quantization algorithm to use 
            bnb_4bit_compute_dtype=torch.bfloat16  # data type of model after quantization
        )      

        print("LoRA activated. Please provide base model.")
        model_path = input()

        print("Loading LoRA model with PEFT...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            local_files_only=False
        )

        model.load_adapter(args.model)
    else:
        print("Loading model...")
        model_path = args.model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                     # Load model in 4-bit mode
            bnb_4bit_use_double_quantization=True, # Nested quantization 
            bnb_4bit_quant_type="nf4",             # Quantization algorithm to use 
            bnb_4bit_compute_dtype=torch.bfloat16  # data type of model after quantization
        )    
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded on {"GPU" if device == "cuda:0" else "CPU"}.")

    while True:
        print("Type prompt or press ENTER to exit:")
        prompt = input()

        if prompt == "":
            print("Exiting...")
            exit(0)
        else:
            parsed_prompt = parse_input(prompt)
            print(generate(model, tokenizer, parsed_prompt))