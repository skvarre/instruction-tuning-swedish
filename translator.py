"""
Translate instruct-data from English to Swedish with GPT-SW3 inference. 

ATTEMPT 1:
Use madlad400-3B for translation.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = '/mnt/pr_SharedNLU/users/tim_olsen/models/models--jbochi--madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = T5Tokenizer.from_pretrained(model_name)



if __name__ == '__main__':

    while True:
        text = input("Enter text to translate: ")    
        # text = """<2sv> Create a narrative for the following situation: Isak asks about the matter. Now Isak is told about the matter."""
        input_ids = tokenizer(f'<2sv> {text}', return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_length=2048, temperature=0.3, do_sample=False)

        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
