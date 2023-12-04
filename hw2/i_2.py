from transformers import pipeline
import json
from tqdm.auto import tqdm
import csv
import datasets
from datasets import load_dataset
data=[]

with open("/home/guest/r12922121/ADL_2023_NTU/hw2/data/public.jsonl", "r") as f:
    train = f.readlines()
    _len = len(train)
    # _len = 5
    for _ in range(_len):
        d = json.loads(train[_])
        data.append(d)

summarizer = pipeline("summarization", model="/home/guest/r12922121/ADL_2023_NTU/hw2/summarization-b4_e4", device=0)
tokenizer = summarizer.tokenizer
model = summarizer.model

with open("sum-beam-3.jsonl", 'w', newline='') as file:
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

with open("sum-topk-3.jsonl", "w") as file:
    for text in tqdm(data):
        summary_3 = summarizer(
            text['maintext'],
            no_repeat_ngram_size=2,
            top_k=35,
            do_sample=True,
            early_stopping=True,
            max_length= 64,
        )
        output = {'title': summary_3[0]['summary_text'], 'id': text['id']}
        json.dump(output, file)
        file.write('\n')

with open("sum-topp-3.jsonl", "w") as file:
    for text in tqdm(data):
        summary_3 = summarizer(
            text['maintext'],
            top_p=0.35,
            do_sample=True,
            early_stopping=True,
            max_length= 64,
        )
        output = {'title': summary_3[0]['summary_text'], 'id': text['id']}
        json.dump(output, file)
        file.write('\n')

# with open("sum-greedy.jsonl", "w") as file:
#     for text in tqdm(data):
#         summary_3 = summarizer(
#             text['maintext'],
#             num_beams=1,
#             do_sample=False,
#             early_stopping=True,
#             max_length= 64,
#         )
#         output = {'title': summary_3[0]['summary_text'], 'id': text['id']}
#         json.dump(output, file)
#         file.write('\n')

with open("sum-temperature-3.jsonl", "w") as file:
    for text in tqdm(data):
        summary_3 = summarizer(
            text['maintext'],
            temperature=0.5,
            do_sample=True,
            early_stopping=True,
            max_length= 64,
        )
        output = {'title': summary_3[0]['summary_text'], 'id': text['id']}
        json.dump(output, file)
        file.write('\n')