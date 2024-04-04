"""
Translate instruct-data from English to Swedish with GPT-SW3 inference. 

ATTEMPT 1:
Use madlad400-3B for translation.

ATTEMPT 2:
GPT-SW3-6.7b-translator
"""

# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import json
from tqdm import tqdm


class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

# Load pre-trained model and tokenizer
model_name = '/mnt/pr_SharedNLU/users/tim_olsen/models/gpt-sw3-6.7b-v2-translator'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=tokenizer.bos_token_id)

def translate(text):
        prompt = f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_length=2048,
            # temperature=0.3,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([stop_on_token_criteria]))

        return tokenizer.decode(outputs[0], skip_special_tokens=False).split("<s> Bot: ")[-1].split("<s>")[0]

# Madlad is a bit weird, so we gotta clean new lines, and then insert them back
def parse(text):
    lines = text.split("\n")
    # translate
    lines = [translate(line) for line in lines]
    return "\n".join([line.strip() for line in lines])

def translate_json(path, output):
    with open (path, "r") as file:
        with open(output, "w") as out:
            for i, line in enumerate(tqdm(file)):
                line = json.loads(line)
                new_dict = {}
                new_dict['instruction'] = parse(line['instruction'])
                new_dict['input'] = parse(line['input']) if line['input'] != '' else ''
                new_dict['output'] = parse(line['output'])
                json.dump(new_dict, out)
                out.write("\n")
                out.flush()


# translate_json("./data/open-instruct-v1.jsonl", "open-instruct-v1-sv.jsonl")

# translate("Create a narrative for the following situation: Isak asks about the matter. Now Isak is told about the matter.")

if __name__ == '__main__':
    while True:
        text = input("Enter text to translate: ")    
        # text = """<2sv> Create a narrative for the following situation: Isak asks about the matter. Now Isak is told about the matter."""
        print(translate(text))
