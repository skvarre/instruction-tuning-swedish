import json
import random
import itertools
import functools
import multiprocessing
import argparse
import os
import random

#import transformers
import sentencepiece
import torch

import indexed_dataset
from transformers import AutoTokenizer

# tokenizer = sentencepiece.SentencePieceProcessor('tokenizer.model')
model_name = "AI-Sweden-Models/gpt-sw3-126m"
tokenizer = AutoTokenizer.from_pretrained(model_name)


ROLEMAP = {'<human>': 'User', 'content': 'User', '<bot>': 'Bot'}
SEP_TOKEN_ID = tokenizer.sep_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id

def fmt_turns_flat(turns):
    """
    This is responsible for formatting turns in the document
    in a FLAT manner.

    turns: [{<human>|<bot>|context : message}]
    out -> str
    """
    ts = []
    for t in turns:
        (role, msg), = t.items()
        ts.append(msg)
    return '\n\n'.join(ts)
    



CHAT_TURN_FORMATS = [
        '{role}\n{msg}',
        '{role}\n{msg}\n',
        '\n{role}\n{msg}',
        '\n{role}\n{msg}\n',
        '{role}: {msg}',
        '{role}: {msg}\n',
        '\n{role}: {msg}',
        '\n{role}: {msg}\n',
        '{role}:\n{msg}',
        '{role}:\n{msg}\n',
        '\n{role}:\n{msg}',
        '\n{role}:\n{msg}\n',
        ]

def fmt_turns_chat(turns):
    """
    This is responsible for formatting turns in the document
    in a CHAT manner.

    turns: [{<human>|<bot>|context : message}]
    out -> [str]
    """
    ret = []
    ts = []
    
    turn_fmt = random.choice(CHAT_TURN_FORMATS)

    for t in turns:
        (role, msg), = t.items()
        ts.append((role, msg))

    groups = itertools.groupby(ts, lambda t: ROLEMAP[t[0]])

    for role, group in groups:
        msg = '\n'.join([c for r, c in group])
        ret.append(turn_fmt.format(role=role, msg=msg))

    return ret

def tokenize_turns_flat(doc, tokenizer, eos_token, sep_token):
    """
    This format flattens the turns and just uses the instructions and responses.
    """
    turns = json.loads(doc)['text']
    ids = [eos_token]

    text = fmt_turns_flat(turns)
    ids += tokenizer.encode(text)

    return torch.tensor(ids, dtype=torch.int64)

def tokenize_turns_chat(doc, tokenizer, eos_token, sep_token, start_with_sep=True):
    """
    This uses fmt_turns and formats the resulting list of turns.

    returns:
    [EOS, *tokens for 1st turn, SEP, *tokens for 2nd turn, SEP, ..., SEP]
    """
    turns = json.loads(doc)['text']
    ids = [eos_token, sep_token]

    for turn in fmt_turns_chat(turns):
        ids += tokenizer.encode(turn)
        ids.append(sep_token)
    return torch.tensor(ids, dtype=torch.int64)

def tokenize_turns(doc, tokenizer, eos_token, sep_token, chat_p=0.8):
    if random.random() <= chat_p:
        return tokenize_turns_chat(doc, tokenizer, eos_token, sep_token)
    else:
        return tokenize_turns_flat(doc, tokenizer, eos_token, sep_token)

def tokenize_chunk(chunk, tokenizer, eos_token, sep_token, chat_p=0.8, both=False):
    agg = []
    for line in chunk:
        if not both:
            agg.append(tokenize_turns(line, tokenizer, eos_token, sep_token, chat_p))
        else:
            agg.append(tokenize_turns(line, tokenizer, eos_token, sep_token, 0.0))
            agg.append(tokenize_turns(line, tokenizer, eos_token, sep_token, 1.0))
    return agg


def line_chunks(path, bytes=1_000_000):
    with open(path, 'rb') as handle:
        while (chunk := handle.readlines(bytes)):
            yield chunk


def tokenize_file(file, out, chat_p, both):

    data_path = indexed_dataset.data_file_path(out)
    idx_path= indexed_dataset.index_file_path(out)
    builder = indexed_dataset.make_builder(data_path, 'mmap', vocab_size=len(tokenizer))

    proc = functools.partial(tokenize_chunk, tokenizer=tokenizer, eos_token=EOS_TOKEN_ID, sep_token=SEP_TOKEN_ID, chat_p=chat_p, both=both)
    chunks = line_chunks(file)

    tensors = itertools.chain.from_iterable(map(proc, chunks))
    
    for tensor in tensors:
        builder.add_item(tensor)
        builder.end_document()

    builder.finalize(idx_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--chat_p', type=float, default=0.8)
    parser.add_argument('--both', action='store_true')
    
    args = parser.parse_args()

    tokenize_file(args.file, os.path.join(args.outdir, os.path.basename(args.file)), args.chat_p, args.both)

