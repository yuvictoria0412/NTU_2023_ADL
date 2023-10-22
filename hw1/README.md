## Environment
```
conda create -n adl-hw1 python=3.9
conda activate adl-hw1
conda install pytorch-1.12.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Multuple choice
**Train**
```
bash train_mc.sh
```

## Question answering
**Train**
```
bash train_qa.sh
```

## Inference
```
python inference.py
```

## Reproduce
```
bash download.sh
bash run.sh adl-hw1/data/context.json adl-hw1/data/test.json qa_pred.csv
```
if you eant