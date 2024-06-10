"""
This is a stupid script to address stupid examples in translated text.
"""
import json
from tqdm import tqdm

""""Heuristic: Remove examples that differ in amount of lines between the original and translated text."""
def line_diff(original_text, translated_text):
    original_text = original_text.split("\n")
    translated_text = translated_text.split("\n")
    return len(original_text) != len(translated_text)

"""Heuristic: Remove examples where the difference in length between the original and translated text is greater than or equal to words."""
def len_difference(original_text, translated_text, threshold):
    original_text = original_text.split()
    translated_text = translated_text.split()
    return abs(len(original_text) - len(translated_text)) >= threshold

"""Heuristic: Remove examples that differ in amount of dots between the original and translated text."""
def dot_diff(original_text, translated_text):
    original_text = original_text.count(".")
    translated_text = translated_text.count(".")
    return original_text != translated_text

def slim_orca_format(path, output):
    with open(path, "r") as file:
        lines = file.readlines()

    with open(output, "w") as out:
        for i, line in enumerate(tqdm((lines))):
            data = json.loads(line)
            conv_list = data['conversations']
            skip = False
            for conv in conv_list:
                if (line_diff(conv['value'][0], conv['value'][1])):
                    skip = True
                    break 
            if skip:
                continue
            data['conversations'] = conv_list
            json.dump(data, out)
            out.write("\n")
            out.flush()

def dpo_format(path, output):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()

    with open(output, "w") as out:
        for i, line in enumerate(tqdm((lines))):
            data = json.loads(line)
            if data['system'] != "\"\"":
                sys = json.loads(data['system'])
                if (line_diff(sys[0], sys[0])):
                    continue
            if line_diff(data['question'][0], data['question'][1]) or \
                line_diff(data['chosen'][0], data['chosen'][1]) or \
                line_diff(data['rejected'][0], data['rejected'][1]):
                continue
            out.write(json.dumps(data))
            out.write("\n")



if __name__ == "__main__":
    path = "./data/Orca-DPO-pairs-geq2k.jsonl"
    output = "./data/Orca-DPO-pairs-geq2k-postprocessed.jsonl"

    dpo_format(path, output)



