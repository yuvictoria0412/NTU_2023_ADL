# HW2


## Environment
```
conda create -n "adl-hw2" python=3.9
conda activate
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/huggingface/accelerate
```
<!-- if you want to run at jupytoer notebook -->
<!-- ![config answer](image.png) -->
<!-- ```
pip install fsspec==2023.9.2    # need this for load_dataset error
pip install ipywidgets  # if you want to use jupyter notebook
``` -->

## Download data
```
gdown https://drive.google.com/uc?id=186ejZVADY16RBfVjzcMcz9bal9L3inXC
unzip data.zip
rm data.zip
```

## Train
```
bash train_sum.sh
```

## Test
```
python inference.py
```

## Evaluation
```
cd ADL23-HW2
python eval.py -r /reference_file.jsonl -s /submission_file.jsonl
```

## Reproduce
```
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```