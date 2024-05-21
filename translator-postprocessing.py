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


if __name__ == "__main__":
    path = "./data/SlimOrca-sv-CONTINUE-2.jsonl"
    output = "./data/SlimOrca-sv-CONTINUE-3.jsonl"

    with open(path, "r") as file:
        lines = file.readlines()

    with open(output, "w") as out:
        for i, line in enumerate(tqdm((lines))):
            data = json.loads(line)
            conv_list = data['conversations']
            skip = False
            for conv in conv_list:
                if (dot_diff(conv['value'][0], conv['value'][1])):
                    skip = True
                    break 
            if skip:
                continue
            data['conversations'] = conv_list
            json.dump(data, out)
            out.write("\n")
            out.flush()
