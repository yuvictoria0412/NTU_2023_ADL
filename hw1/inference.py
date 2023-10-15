import torch
import transformers
import json
import csv
from tqdm.auto import tqdm
from transformers import pipeline
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

with open("data/context.json", "r") as f:
    context_file = json.load(f)
with open("data/test.json", "r") as f:
    test_file = json.load(f)
file = open("eight_try_r2_4.csv", 'w', newline='')
output_file = csv.writer(file)
output_file.writerow(["id","answer"])


## multiple choice
tokenizer = AutoTokenizer.from_pretrained("/home/guest/r12922121/ADL_2023_NTU/hw1/output/output_hf1_e3")
model = AutoModelForMultipleChoice.from_pretrained("/home/guest/r12922121/ADL_2023_NTU/hw1/output/output_hf1_e3")

prompt = []
candidate = []
for ques in test_file:
    prompt.append(ques["question"])
    context = []
    for para in ques["paragraphs"]:
        context.append(context_file[para])
    candidate.append(context)
# it = len(prompt)
it = 10
prediction = []
for i in tqdm(range(it)):
    input = []
    for j in range(4):
        input.append([prompt[i], candidate[i][j]])
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = torch.tensor(0).unsqueeze(0)
    outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    # print(candidate[i][predicted_class])
    prediction.append(candidate[i][predicted_class])

## question answer
from transformers import pipeline

question_answerer = pipeline("question-answering", model="/home/guest/r12922121/ADL_2023_NTU/hw1/sample_output_7_newtrain")
for i in tqdm(range(it)):
    str = question_answerer(question=prompt[i], context=prediction[i])["answer"]
    # print([test_file[i]["id"], str])
    output_file.writerow([test_file[i]["id"], str])

file.close()