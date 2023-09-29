# ADL 2023 hw1
## Steps
1. download code *run_swag_no_trainer.py* from [huggingface](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py)
2. download *requirements.txt* and *run_no_trainer.sh*
3. command
```
pip install -r requirements.txt
```
4. in file *run.sh* add
``` 
pip install -U git+https://github.com/huggingface/accelerate
```
5. add to file *requirements.txt*
```
transformers == 4.22.2
torch == 1.12.1
```
6. 
``` 
bash run.sh
```

**bold**
* Italic *  
> blockquote


`code`
---
link:
[title](http://)
Image ![alt text](image.jpg)