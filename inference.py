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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import AutoPeftModelForCausalLM
import argparse
import torch 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048

class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

def parse_translation(prompt):
    return f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{prompt}<s>Bot:"

#TODO: Hardcoded. Assumes bos_token = <s> and eos_token = <|endoftext|>.
def parse_input(system_prompt, prompt):
    # parsed_output = f"\n<s>User\n{prompt}\n<s>Bot\n"
    # return f"<|endoftext|>{parsed_output}" if beginning_of_conversation else parsed_output
    if system_prompt:
        return f"<|endoftext|><s>\nSYSTEM\n{system_prompt}\n\n<s>USER:\n{prompt}\n<s>ASSISTANT:\n"
    else:
        return f"<|endoftext|><s>\nUSER:\n{prompt}\n<s>ASSISTANT:\n"
    
def generate_translation(model, tokenizer, prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    dynamic_max_length = MAX_LENGTH - inputs.shape[1]
    outputs = model.generate(
        inputs,
        max_length=dynamic_max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return output.split('<s>Bot: ')[-1]

def generate(model, tokenizer, prompt):

    # messages = [
    # {"role": "system", "content": "Du är en AI-assistent. Förklara om meningen är grammatiskt korrekt."},
    # {"role": "user", "content": "Jag har ett badboll."},
    # ]

    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # ).to(device)

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    dynamic_max_length = MAX_LENGTH - inputs.shape[1]
    outputs = model.generate(
        inputs,
        max_length=dynamic_max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Stop at the first <s> token.
    # return "".join(output.split('<s> ASSISTANT:'))
    if args.nochat:
        return output
    else:
        return output.split('<s>')[-2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/results")
    parser.add_argument('--lora', action='store_true', help="Whether to use LoRA for inference. This assumes adapters as model argument. Default is False.")
    parser.add_argument('--translate', action='store_true', help="For inference on gpt-sw3-translator")
    parser.add_argument('--nochat', action='store_true', help="Skip Chat Template")

    parser.set_defaults(nochat=False)
    parser.set_defaults(lora=False)
    parser.set_defaults(translate=False)


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
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     #quantization_config=quantization_config, 
                                                     torch_dtype=torch.bfloat16
                                                     ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=tokenizer.bos_token_id)

    print(f"Model loaded on {"GPU" if device == "cuda:0" else "CPU"}.")
    print("Enter system prompt? [Y/n]")
    system_prompt = input()
    if system_prompt.lower() == "y":
        print("Enter system prompt:")
        system_prompt = input()
        print("System prompt can be changed by typing SYS")
    elif system_prompt.lower() == "n":
        print("Skipping system prompt.")
        system_prompt=None
    else:
        print("Invalid input. Skipping system prompt.")
        system_prompt = None
    while True:

        print("Type prompt: ([press ENTER to exit], [type SYS to change System prompt], [type --file to read from file])")
        prompt = input()

        if prompt == "":
            print("Exiting...")
            exit(0)
        elif prompt.lower() == "sys":
            print("Enter system prompt: (Leave blank to skip system prompt)\nCurrent system prompt is: ", system_prompt if system_prompt else "[None]")
            system_prompt = input()
            parsed_prompt = parse_input(system_prompt, prompt)
        else:
            if prompt == "--file":
                print("Enter file path:")
                file_path = input()
                with open(file_path, 'r') as f:
                    prompt = f.read()

            if args.translate:
                parsed_prompt = parse_translation(prompt)
                print(generate_translation(model, tokenizer, parsed_prompt))
            else:
                if not args.nochat:
                    parsed_prompt = parse_input(system_prompt, prompt)
                    print(generate(model, tokenizer, parsed_prompt))
                else: 
                    print(generate(model, tokenizer, prompt))