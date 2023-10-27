# HW2
_hi🙋🏻‍♀️ i am Victoria_

## Environment
```
conda create -n "adl-hw2" python=3.9
conda activate
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/huggingface/accelerate
accelerate config
```
![config answer](image.png)
if you want to run at jupytoer notebook
```
pip install fsspec==2023.9.2    # need this for load_dataset error
pip install ipywidgets  # if you want to use jupyter notebook
```

## Access data
```
gdown https://drive.google.com/uc?id=186ejZVADY16RBfVjzcMcz9bal9L3inXC
unzip data.zip
rm data.zip
```

## Evaluation
```
git clone https://github.com/moooooser999/ADL23-HW2.git
cd ADL23-HW2
pip install -e tw_rouge
```