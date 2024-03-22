from transformers import GenerationConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
import datasets
from utils import get_prompt, get_bnb_config
import json
from  tqdm import tqdm
import sys

base_model_path = sys.argv[1]
peft_path = sys.argv[2]
data_path = sys.argv[3]
output_file_path = sys.argv[4]

# base_model_path='/home/guest/r12922121/ADL_2023_NTU/hw3/Taiwan-LLM-7B-v2.0-chat'
# peft_path='/home/guest/r12922121/ADL_2023_NTU/hw3/c'
# data_path='/home/guest/r12922121/ADL_2023_NTU/hw3/data/public_test.json'

output_file = open(output_file_path, 'w', encoding='utf-8')

with open(data_path, "r") as f:
    data = json.load(f)
data = data[:10]
sequences = [d['instruction'] for d in data]
sequences = [get_prompt(s) for s in sequences]

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if base_model_path:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map = 'auto',
        quantization_config=get_bnb_config(),
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# model = PeftModel.from_pretrained(model, peft_path)
res_l = []
for i in tqdm(range(len(sequences))):

    # inputs = tokenizer(sequences[i], padding=True, return_tensors="pt").to('cuda')
    inputs = tokenizer(sequences[i], return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=512,
            top_p=0.9,
            temperature=1e-8,
        )
    )
    # output = [(tokenizer.decode(output, skip_special_tokens=True)) for output in outputs]
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)


    l = len(sequences[i])
    res = {
        "id": data[i]["id"],
        "output": output[l+1:]}
    res_l.append(res)

json.dump(res_l, output_file,  ensure_ascii=False, indent=4)
output_file.close()

