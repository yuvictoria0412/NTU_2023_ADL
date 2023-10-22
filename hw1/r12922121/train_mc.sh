python "train_mc.py"\
    --model_name_or_path "hfl/chinese-roberta-wwm-ext"\
    --train_file "$1"\
    --validation_file "$2"\
    --context_file "$3"\
    --max_seq_length 512\
    --per_device_train_batch_size 1\
    --gradient_accumulation_steps 2\
    --num_train_epochs 1\
    --learning_rate 3e-5\
    --output_dir "$4"\

