"""
TESTING OF SPELLCHECKER.

Spellcheck the data based on a given threshold. 
The threshold is the ratio of correctly spelled words to the total number of words in the example.
"""
from spylls.hunspell import Dictionary
import json
import re

dictionary = Dictionary.from_files('sv_SE')
path = "data/2_examples.jsonl"

def score(turns):
    total = 0
    score = 0
    for turn in turns:
        for _, msg in turn.items():
            for words in re.split(r'[ ,!?:;"]+' ,msg):
                total += 1
                # print(words, dictionary.lookup(words))
                score += 1 if dictionary.lookup(words) else 0
    print(score, total, score/total)
    return score/total

def spellcheck(turns, threshold):
    return True if score(turns) > threshold else False

#TODO: Implement a new dataset based on the spellchecked data.
def spellchecked_data(path, threshold):
    pass

if __name__ == '__main__':
    with open (path, "r") as file:
        for line in file:
            line = json.loads(line)
            turns = line['text']
            print(spellcheck(turns, 0.35))