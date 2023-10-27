python run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file /home/guest/r12922121/ADL_2023_NTU/hw2/data/train.jsonl \
    --validation_file /home/guest/r12922121/ADL_2023_NTU/hw2/data/public.jsonl \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir summarization-3 \
    --text_column maintext \
    --summary_column title \
    --max_source_length 256 \
    --max_target_length 64 \
    --pad_to_max_length \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-4 \
    --num_train_epochs  4\
    --gradient_accumulation_steps 4 \
    
    # --max_train_samples 10 \
    # --max_eval_samples 10 \


# accelerate launch run_summarization_no_trainer.py \
#     --model_name_or_path t5-small \
#     --train_file /home/guest/r12922121/ADL_2023_NTU/hw2/data/train.jsonl \
#     --validation_file /home/guest/r12922121/ADL_2023_NTU/hw2/data/public.jsonl \
#     --dataset_config "3.0.0" \
#     --source_prefix "summarize: " \
#     --output_dir summarization-1 \
#     --text_column maintext \
#     --summary_column title \