python run_qa_no_trainer.py \
  --model_name_or_path "hfl/chinese-macbert-large" \
  --train_file "$1" \
  --validation_file "$2" \
  --context_file "$3"\
  --max_seq_length 512 \
  --output_dir "$4" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --with_tracking \

