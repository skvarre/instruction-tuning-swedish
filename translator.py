"""
Translate instruct-data from English to Swedish with GPT-SW3 inference. 
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
# from transformers import AutoProcessor, SeamlessM4TModel
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, pipeline
import torch
import json
from tqdm import tqdm


class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

# Load pre-trained model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = './merged-models/gpt-sw3-6.7b-v2-translator-5/'
# processor = AutoProcessor.from_pretrained(model_name)
# model = SeamlessM4TModel.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline(
    task="text-generation",
    model = model_name,
    device=device,
    torch_dtype=torch.bfloat16,
    batch_size=8
)
stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=pipe.tokenizer.bos_token_id)

"""
Translate a single text from English to Swedish.
Assumes GPT-SW3-6.7b-translator
"""
def translate(text):
        prompt = f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:"
        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        dynamic_max_length = 2048 - input_ids.shape[1]
        outputs = pipe(
            prompt,
            max_length=dynamic_max_length,
            # temperature=0.3,
            truncation=True,
            stopping_criteria=StoppingCriteriaList([stop_on_token_criteria]))

        return outputs[0]['generated_text'].split("<s>Bot: ")[-1]

# def translate(text):
#     input_ids = tokenizer(f"<2sv> {text}", return_tensors="pt").input_ids.to(model.device)
#     outputs = model.generate(input_ids=input_ids, max_length=128)

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


"""
Remove newlines, translates text, and appends the newlines back.

Newlines has proven to be a problem with google/madlad-400-3b.
"""
def parse(text):
    lines = text.split("\n")
    # translate
    lines = [translate(line) for line in lines]
    return "\n".join([line.strip() for line in lines])

"""
Translate jsonl file with instruction data from English to Swedish.

Assumes format:
{
    "instruction",
    "input",
    "output"
}
"""
def translate_json_instruct(path, output):
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

def save_line(path, line):
    with open(path, "w") as out:
        json.dump(line, out)

def process_data(conv_list, keep_original=False):
    for conv in conv_list:
        if keep_original:
            new_conv = []
        # Avoid translating if token length is too long before translation.
        if len(pipe.tokenizer.encode((conv['value']))) <= 2048:
            try:
                if keep_original:
                    new_conv.append(conv['value'])
                    new_conv.append(translate(conv['value']))
                    conv['value'] = new_conv
                else:
                    conv['value'] = translate(conv['value']) 
            except ValueError as e:
                return None
        else:
            return None
    return conv_list

"""
Assumes Conversational format of dataset.
"""
def translate_json(path, output, keep_original=False):
    latest_line = 40000 #
    with open(path, "r") as file:
        lines = file.readlines()
    
    with open(output, "w" if latest_line == 0 else "a") as out:
        for _, line in enumerate(tqdm(lines[latest_line:], initial=latest_line, total=len(lines[latest_line:]))):
            data = json.loads(line)
            conv_list = process_data(data['conversations'], keep_original=keep_original)
            if conv_list is None:
                continue
            data['conversations'] = conv_list 
            json.dump(data, out)
            out.write("\n")
            out.flush()

def translate_sv_en(path, output):
    latest_line = 53 #
    with open(path, "r") as file:
        lines = file.readlines()
    
    with open(output, "w" if latest_line == 0 else "a") as out:
        for _, line in enumerate(tqdm(lines[latest_line:], initial=latest_line, total=len(lines[latest_line:]))):
            data = json.loads(line)
            conv_list = data['conversations']
            dct = {
                "en":"", 
                "sv":"",
            }
            for conv in conv_list:
                # Avoid translating if token length is too long before translation.
                if len(pipe.tokenizer.encode((conv['value']))) <= 2048:
                    try:
                        dct['sv'] = translate(conv['value'])
                        dct['en'] = conv['value']
                        out.write("\n")
                        out.flush()
                        json.dump(dct, out)
                    except ValueError as e:
                        break

def process_dpo(conv_list, keep_original=False):
    for pair in conv_list:
        if keep_original:
            new_conv = []

        if conv_list[pair] == "":
            continue

        # Avoid translating if token length is too long before translation.
        if len(pipe.tokenizer.encode((conv_list[pair]))) <= 2048:
            try:
                if keep_original:
                    new_conv.append(conv_list[pair])
                    new_conv.append(translate(conv_list[pair]))
                    conv_list[pair] = new_conv
                else:
                    conv_list[pair] = translate(conv_list[pair]) 
            except ValueError as e:
                return None
        else:
            return None
    return conv_list

def transalte_dpo(path, output, keep_original=False):
    latest_line = 0

    with open(path, "r") as file:
        lines = file.readlines()
    
    with open(output, "w" if latest_line == 0 else "a") as out:
        for _, line in enumerate(tqdm(lines[latest_line:], initial=latest_line, total=len(lines[latest_line:]))):
            data = json.loads(line)
            conv_list = process_dpo(data, keep_original=keep_original)
            if conv_list is None:
                continue
            data = conv_list 
            json.dump(data, out)
            out.write("\n")
            out.flush()
                

transalte_dpo("./one.jsonl", "./one-sv.jsonl", keep_original=True)

# if __name__ == '__main__':
#     while True:
#         text = input("Enter text to translate: ")    
#         # text = """<2sv> Create a narrative for the following situation: Isak asks about the matter. Now Isak is told about the matter."""
#         print(translate(text))
