export CUDA_VISIBLE_DEVICES=8
python train.py \
    --base_model_path ziqingyang/visualcla-7b-v0.1 \
    --lora_r=4 \
    --lora_alpha=128 \
    --lora_dropout=0.02 \
    --num_train_epochs=3 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_steps=10 \
    --output_dir="./d" \