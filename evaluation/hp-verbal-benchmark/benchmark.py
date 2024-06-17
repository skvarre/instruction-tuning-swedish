import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import json
import random
from enums import AI_SWE_GPTSW3, CUSTOM_GPTSW3
from tqdm import tqdm 
from prompts import ORD_prompt, LÄS_prompt, MEK_prompt, extract_few_shot
from scores import judge_answer, calculate_all
import datetime

MAX_LENGTH = 2048
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

def generate(model, tokenizer, prompt, stop_on_token_criteria, mapper):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs,
        max_length=MAX_LENGTH,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
    )

    output = tokenizer.decode(outputs[0], skip_special_tokens=False).split(f"{mapper["assistant"]}\n")[-1].split('<s>')[0]
    return output


def parse_data(data):
    random.shuffle(data)
    data_lists = [[],[],[]]
    for d in data:
        if d['test'] == "ORD":
            data_lists[0].append(d)
        elif d['test'] == "LÄS":
            data_lists[1].append(d)
        elif d['test'] == "MEK":
            data_lists[2].append(d)
    return data_lists

def output_results(results, model_path):
    output_path = "hp-benchmark-results.jsonl"
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["date"] = date
    results["model"] = model_path.split("/")[-1]

    with open(output_path, "a") as f:
        f.write(json.dumps(results) + "\n")


def benchmark_model(model_path):
    n_shot = 5
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 torch_dtype = dtype).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stop_on_token_criteria = StopOnTokenCriteria(stop_token_id=tokenizer.bos_token_id)
    bos_token = CUSTOM_GPTSW3.bos_token.value
    eos_token = CUSTOM_GPTSW3.eos_token.value
    mapper = CUSTOM_GPTSW3.mapper.value

    prompts = {"ord": ORD_prompt, "läs": LÄS_prompt, "mek": MEK_prompt}
    tasks = ["ord", "läs", "mek"]

    # Load data. Also available at hf: skvarre/hogskoleprovet-ORD-LAS-MEK
    path = "hogskoleprovet-ORD-LAS-MEK.jsonl"
    with open(path, "r") as f:
        data = f.readlines()
    data = parse_data([json.loads(d) for d in data])
    data = {t : data[i] for i, t in enumerate(tasks)}

    # Benchmark model
    print("Preparing to benchmark model on Högskoleprovet: ORD, LÄS, MEK.")

    results = {"ord": [], "läs": [], "mek": []}
    for t in tasks:
        print(f"Currently benchmarking model on {t.upper()}.")
        if n_shot > 0 and t in ["ord", "mek"]:
            """We have to perform zero-shot on LÄS due to context length constraints."""
            few_shot_examples = extract_few_shot(data[t][:n_shot], t, eos_token, bos_token, mapper)
        else:
            """If we are zero-shotting the model, we instead provide the beginning tokens."""
            few_shot_examples = f"{eos_token}{bos_token}\n"
        benchmark_data = data[t][n_shot:]

        if t == "ord":
            continue 

        for i in tqdm(range(10)):
            sub_results = []
            for _, example in tqdm(enumerate(benchmark_data), total=len(benchmark_data)):
                prompt = f"{few_shot_examples}\n{prompts[t](example, bos_token, mapper)}"
                if tokenizer.encode(prompt, return_tensors="pt").to(DEVICE).shape[1] <= MAX_LENGTH:
                    print("lol")
                    output = generate(model, tokenizer, prompt, stop_on_token_criteria, mapper)
                    sub_results.append(judge_answer(example, output))
            results[t].append(sub_results)
    
    final_results = calculate_all(results, model_path)
    output_results(final_results)
    
