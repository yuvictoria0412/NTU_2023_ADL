#   --dataset_name squad \

#   --train_file data/train.json \
#   --validation_file data/valid.json \


# sample input
python run_qa_no_trainer.py \
  --model_name_or_path hfl/chinese-macbert-base \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir sample_output/ \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \