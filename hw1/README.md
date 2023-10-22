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
bash train_mc.sh /path/to/train.json /path/to/valid.json /path/to/context.json /path/to/output_file
```

## Question answering
**Train**
```
bash train_qa.sh /path/to/train.json /path/to/valid.json /path/to/context.json /path/to/output_file
```

## Inference
```
python inference.py /path/to/context.json /path/to/test.json /path/to/output_file /path/to/mc_model_dir /path/to/qa_model_dir
```

## Reproduce
```
bash download.sh
bash run.sh /path/to/context.json /path/to/test.json /path/to/output_file
```
