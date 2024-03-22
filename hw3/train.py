
#########################################
# reference:
#   https://medium.com/@newhardwarefound/qlora-with-llama-2-ca1b4bcf26f0
# https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
#
#########################################

import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
import datasets
from utils import get_prompt, get_bnb_config
from dataclasses import dataclass, field
import transformers
import copy
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
''' for bonus'''
from huggingface_hub import login

login("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")
''' end bonus '''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/guest/r12922121/ADL_2023_NTU/hw3/Taiwan-LLM-7B-v2.0-chat",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="/home/guest/r12922121/ADL_2023_NTU/hw3/data/train.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,
    )
    args = parser.parse_args()
    return args


def create_peft_config(modules, args):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=args.lora_r, #64,  # dimension of the updated matrices
        lora_alpha=args.lora_alpha, #16,  # parameter for scaling
        target_modules=modules,
        lora_dropout=args.lora_dropout,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model_and_tokenizer(base_model_path):
    """Load the model and tokenizer."""
    
    print("create bnb config")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map = 'auto',
        quantization_config=get_bnb_config(),
    )
    print("create tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    '''
    here they are different
    '''
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    return model, tokenizer

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

IGNORE_INDEX = -100
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['instruction']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def main():
    print("Parsing arguments...")
    args = parse_args()
    print(args)
    # Load dataset
    print("Loading dataset...")
    # datasets.logging.set_verbosity_info()
    datasets.logging.enable_progress_bar()
    data_file = {"train": args.train_data}
    dataset = load_dataset("json", data_files=data_file)

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.base_model_path)
    print("Preparing model for k-bit training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print("create peft model...")
    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules, args)
    model = get_peft_model(model, peft_config)
    print("Got model")

    print_trainable_parameters(model)
    
    print("Data collator setup...")
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=280,
        target_max_len=512,
        train_on_source=False,
        predict_with_generate=False,
    )
    
    print("Trainer setup...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=2,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=100,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            num_train_epochs=args.num_train_epochs,
            gradient_checkpointing=False,
            logging_dir=f"{args.output_dir}/logs",
            logging_strategy="steps",
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=280,
            target_max_len=512,
            train_on_source=False,
            predict_with_generate=False),
    )

    model.config.use_cache = False

    print("check model type")
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
    do_train = True

    
    if do_train:
        print("Start training...")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # print(metrics)    
        f.write(str(metrics)+'\n')
    else:
        print("model type error")
        exit(0)

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    f = open('train_loss_per_step.txt', 'a')

    main()
    