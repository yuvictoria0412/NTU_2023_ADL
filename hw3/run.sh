export CUDA_VISIBLE_DEVICES=8

python ppl.py \
    --base_model_path /home/guest/r12922121/ADL_2023_NTU/hw3/Taiwan-LLM-7B-v2.0-chat \
    --test_data_path /home/guest/r12922121/ADL_2023_NTU/hw3/data/private_test.json \
    --peft_path /home/guest/r12922121/ADL_2023_NTU/hw3/c
