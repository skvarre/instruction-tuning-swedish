def extract_few_shot(data, task, eos_token, bos_token, mapper) -> str:
    answer_mapper = {"A":0, "B":1, "C":2, "D":3, "E":4}
    beginning_tokens = f"{eos_token}{bos_token}\n"
    if task == "ord":
        pre_prompt = f"{mapper['user']}\nDu kommer att få ett ord. Välj bland alternativen det ord som bäst beskriver det givna ordet."
        return beginning_tokens + '\n'.join([
                f"{pre_prompt}\nOrd: {d['question']}\n\nAlternativ:\n{alt_string}\n{bos_token}\n{mapper['assistant']}\n{d['options'][answer_mapper[d['answer']]]}\n{bos_token}"
                for d in data
                for alt_string in ['\n'.join(d['options'])]
            ])        
    elif task == "läs":
        pre_prompt = f"{mapper['user']}\nLäs texten nedan och svara på frågan som följer. Välj svar utifrån svarsalternativen."
        return beginning_tokens + '\n'.join([
            f"{pre_prompt}\n\n{d['context']}\n\nFråga:\n{d['question']}\n\nAlternativ:\n{alt_string}\n{bos_token}\n{mapper['assistant']}\n{d['options'][answer_mapper[d['answer']]]}\n{bos_token}"
            for d in data
            for alt_string in ['\n'.join(d['options'])]
        ])
    elif task == "mek":
        pre_prompt = f"{mapper['user']}\nDu kommer att få en text med en eller flera luckor i form av understreck. Välj bland alternativen det eller de ord som bäst fyller i luckan/luckorna."
        return beginning_tokens + '\n'.join([
            f"{pre_prompt}\n\nText:\n{d['question']}\n\nAlternativ:\n{alt_string}\n{bos_token}\n{mapper['assistant']}\n{d['options'][answer_mapper[d['answer']]]}\n{bos_token}"
            for d in data
            for alt_string in ['\n'.join(d['options'])]
        ])

def LÄS_prompt(example, bos_token, mapper) -> str:
    pre_prompt = f"{mapper['user']}\nLäs texten nedan och svara på frågan som följer. Välj svar utifrån svarsalternativen."
    alternatives = "\n".join(opt for opt in example['options'])
    return f"{pre_prompt}\n\n{example['context']}\n\nFråga:\n{example['question']}\n\nAlternativ:\n{alternatives}\n{bos_token}\n{mapper['assistant']}\n"

def ORD_prompt(example, bos_token, mapper) -> str:
    pre_prompt = f"{mapper['user']}\nDu kommer att få ett ord. Välj bland alternativen det ord som bäst beskriver det givna ordet."
    alternatives = "\n".join(opt for opt in example['options'])
    return f"{pre_prompt}\nOrd: {example['question']}\n\nAlternativ:\n{alternatives}\n{bos_token}\n{mapper['assistant']}\n" 

def MEK_prompt(example, bos_token, mapper) -> str:
    pre_prompt = f"{mapper['user']}\nDu kommer att få en text med en eller flera luckor i form av understreck. Välj bland alternativen det eller de ord som bäst fyller i luckan/luckorna."
    alternatives = "\n".join(opt for opt in example['options'])
    return f"{pre_prompt}\n\nText:\n{example['question']}\n\nAlternativ:\n{alternatives}\n{bos_token}\n{mapper['assistant']}\n"