# ADL 2023 hw1
## Preparation
1. install *miniconda*
2. conda create -n [*env_name*] python = [*python version*]
3. conda activate [*env_name*]

## Steps
1. download folder *multiple-choice* from [huggingface](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)
2. add to file *requirements.txt*:
```
transformers >= 4.33.3
scikit-learn >= 1.1.2 
```
3. in file *run_swag_no_trainer.py*, change the min version:
```
check_min_version("4.33.0")
```
4. in file *run.sh* add
``` 
pip install -U -r "requirements.txt"
pip install -U git+https://github.com/huggingface/accelerate
```
5. 
``` 
bash run.sh
```


### markdown language
**bold**
* Italic *  
> blockquote
`code`
---
link:
[title](http://)
Image ![alt text](image.jpg)