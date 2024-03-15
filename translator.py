"""
Translate instruct-data from English to Swedish with GPT-SW3 inference. 

ATTEMPT 1:
Use madlad400-3B for translation.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json
from tqdm import tqdm

# Load pre-trained model and tokenizer
model_name = '/mnt/pr_SharedNLU/users/tim_olsen/models/models--jbochi--madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def translate(text):
        input_ids = tokenizer(f'<2sv> {text}', return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_length=2048, temperature=0.3, do_sample=False)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Madlad is a bit weird, so we gotta clean new lines, and then insert them back
def parse(text):
    lines = text.split("\n")
    # translate
    lines = [translate(line) for line in lines]
    return "\n".join([line.strip() for line in lines])

def translate_json(path, output):
    with open (path, "r") as file:
        with open(output, "w") as out:
            for _, line in enumerate(tqdm(file)):
                line = json.loads(line)
                new_dict = {}
                new_dict['instruction'] = parse(line['instruction'])
                new_dict['input'] = parse(line['input']) if line['input'] != '' else ''
                new_dict['output'] = parse(line['output'])
                json.dump(new_dict, out)
                out.write("\n")
                out.flush()


translate_json("./data/open-instruct-v1.jsonl", "open-instruct-v1-sv.jsonl")

# if __name__ == '__main__':
#     while True:
#         text = input("Enter text to translate: ")    
#         # text = """<2sv> Create a narrative for the following situation: Isak asks about the matter. Now Isak is told about the matter."""
#         input_ids = tokenizer(f'<2sv> {text}', return_tensors="pt").input_ids.to(model.device)
#         outputs = model.generate(input_ids=input_ids, max_length=2048, temperature=0.3, do_sample=False)

#         print(tokenizer.decode(outputs[0], skip_special_tokens=True))
