from transformers import pipeline
import json
from tqdm.auto import tqdm
import csv
import datasets
from datasets import load_dataset
import sys


test_file_dir = sys.argv[1]
output_file_dir = sys.argv[2]
model_dir = sys.argv[3]


data=[]
with open(test_file_dir, "r") as f:
    train = f.readlines()
    _len = len(train)
    for _ in range(_len):
        d = json.loads(train[_])
        data.append(d)

summarizer = pipeline("summarization", model=model_dir, device=0)

with open(output_file_dir, 'w', newline='') as file:
    for text in tqdm(data):
        str = summarizer(
            text['maintext'],
            no_repeat_ngram_size=2,
           num_beams=8,
            do_sample=False,
            early_stopping=True,
        )
        output = {'title': str[0]['summary_text'], 'id': text['id']}
        json.dump(output, file)
        file.write('\n')
