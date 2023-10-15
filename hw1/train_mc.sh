# pip install -r "requirements.txt"
# pip install -U git+https://github.com/huggingface/accelerate

# export DATASET_NAME=swag

# python run_swag_no_trainer.py \
#   --model_name_or_path bert-base-cased \
#   --dataset_name $DATASET_NAME \
#   --max_seq_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 1 \
#   --output_dir /tmp/$DATASET_NAME/

# python run_swag_no_trainer.py --train_file ./data/train.json --validation_file ./data/valid.json --context_file ./data/context.json --model_name_or_path bert-base-chinese --max_seq_length 512 --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --num_train_epochs 1 --learning_rate 3e-5 --output_dir ~/ADL/output-10-7-1/ 

python train_mc.py\
    --model_name_or_path "hfl/chinese-roberta-wwm-ext-large"\
    --train_file "data/train.json"\
    --validation_file "data/valid.json"\
    --context_file "data/context.json"\
    --max_seq_length 512\
    --per_device_train_batch_size 1\
    --gradient_accumulation_steps 2\
    --num_train_epochs 3\
    --learning_rate 3e-5\
    --output_dir "output/output_roberta_e3_large_10/"\
    
    --max_train_steps 10\


# python run_mc.py \
#     --model_name_or_path /home/guest/r12922121/ADL_2023_NTU/hw1/output/output_hf1_1 \
#     --test_file data/test.json \
#     --do_predict\
#     --learning_rate 5e-5 \
#     --num_train_epochs 3 \
#     --output_dir output_mc/test_mc_1 \
#     --output_file ./result/predictions_mc_1.json\
#     --per_device_eval_batch_size=16 \
#     --per_device_train_batch_size=16 \
#     --overwrite_output\
#     --max_seq_length 512\
#     --pad_to_max_length