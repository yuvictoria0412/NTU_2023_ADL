import numpy as np
import pandas as pd

import logging 
import math
import os
import datetime

from dataclasses import dataclass, field
from typing import Optional, Union


import datasets
from datasets import load_dataset


'''
AutoConfig:
AutoConfig 是Hugging Face Transformers庫中的一個類別，用於自動選擇合適的模型配置。它根據提供的模型名稱自動選擇該模型所需的配置，無需手動指定。

AutoModelForMultipleChoice:
AutoModelForMultipleChoice 也是Hugging Face Transformers庫中的一個類別，它用於多選擇問題的預訓練或微調。它可以根據提供的模型名稱自動選擇相應的預訓練模型，並允許對其進行微調以進行多選擇問題的預測。

AutoTokenizer:
AutoTokenizer 是Hugging Face Transformers庫中的一個類別，用於自動選擇合適的分詞器。它根據提供的模型名稱自動選擇該模型所需的分詞器，無需手動指定。

HfArgumentParser:
HfArgumentParser 是Hugging Face Transformers庫中的一個工具，用於解析和處理命令行參數。它簡化了訓練和微調Transformer模型時的參數設置和管理。

Trainer:
Trainer 是Hugging Face Transformers庫中的一個類別，用於訓練和微調Transformer模型。它提供了一個高級的訓練界面，允許用戶輕鬆設置訓練過程，包括數據載入、損失計算、模型更新等。

TrainingArguments:
TrainingArguments 是Hugging Face Transformers庫中的一個類別，用於定義訓練過程的參數和設置。它用於配置模型訓練的各種參數，如學習速率、批次大小、訓練輪數等。

default_data_collator:
default_data_collator 是一個預設的數據整合函數，用於將輸入數據轉換為模型可以處理的格式。它在訓練過程中用於處理數據載入和批次處理。

set_seed:
set_seed 是一個用於設置隨機種子的函數。在訓練和微調Transformer模型時，通常需要固定隨機種子以確保實驗的可重現性。這個函數可以用於設置隨機種子的值。
'''
from transformers import (AutoConfig,
                          AutoModelForMultipleChoice,
                          AutoTokenizer,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          default_data_collator,
                          set_seed)





''' i am a  divider '''
@dataclass
class ModelArguments:
    ''' fine-tune arguments '''
    
    model_name_or_path: str = field(
        default= None,
        metadata= {
            "help": "Path to pretrained model from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default= None,
        metadata= {
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default= None,
        metadata= {
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    cache_dir: Optional[str] = field(
        default= './cache',
        metadata= {
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        }
    )
    use_fast_tokenizer: bool = field(
        default= True,
        metadata= {
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
            }
    )
    use_auth_token: bool = field(
        default= False,
        metadata={
            "help":
            ("Will use the token generated when running `huggingface-cli login` (necessary to use this script "
             "with private models).")
        },
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={'help': 'testing data'},
    )
    context_file: str = field(
        default=None,
        metadata={'help': 'context data'},
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={'help': 'output file'},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=6,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in [
                "csv", "json"], "`validation_file` should be a csv or a json file."

def main():
    ''' 1. Parser '''
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ''' 2. logger '''
    logger = logging.getLogger(__name__)    # current module name
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # file handler
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_filename = f'log_{today_date}.log'
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Set the level to control which messages get printed
    logger.addHandler(console_handler)

    logger.info("logger initialized successed")

    ''' 3. seed '''
    if training_args.seed is not None:
        set_seed(training_args.seed)

    ''' 4. Load Data '''
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    # only_raw_train = load_dataset( "json",data_files=data_args.train_file)
    # print(only_raw_train)
    raw_train_data = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=model_args.cache_dir)
    
    # print(raw_train_data)

    ''' what is down here? '''
    # if raw_train_data["train"] is not None:
    #     column_names = raw_train_data["train"].column_names
    # else:
    #     column_names = raw_train_data["validation"].column_names

    ''' 5. load config from Transformer'''
    # if model_args.config_name:
    #     config = AutoConfig.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir = model_args.cache_dir
    #         )
    # elif model_args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir = model_args.cache_dir
    #         )
    # else:
    #     logger.debug("can't load config from Transformer")

    
    ''' 6. load tokenizer from Transformer'''
    if model_args.tokenizer_name is not None: # 若有輸入 tokenizer 的名子
        logger.info("tokenizer_name")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            use_fast= model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path is not None:   # 若有輸入 model name or path (從 hugging face 那裡)
        logger.info("model_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            use_fast= model_args.use_fast_tokenizer
        )
    else:
        logger.info("can't load tokenizer from Transformer")
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    ''' 7. load model from Transformer'''
    # if model_args.model_name_or_path:
    #     logger.info("you are using model: " + model_args.model_name_or_path)
    #     model = AutoModelForMultipleChoice.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir= model_args.cache_dir,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         # trust_remote_code=model_args.trust_remote_code,
    #     )
    # else:
    #     logger.debug("didn't use model from transfromer")
    

if __name__ == "__main__":
    main()