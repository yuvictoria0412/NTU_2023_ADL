export CUDA_VISIBLE_DEVICES=7

python inference.py \
    /home/guest/r12922121/ADL_2023_NTU/hw3/Taiwan-LLM-7B-v2.0-chat \
    '/home/guest/r12922121/ADL_2023_NTU/hw3/c' \
    '/home/guest/r12922121/ADL_2023_NTU/hw3/data/private_test.json' \
    '/home/guest/r12922121/ADL_2023_NTU/hw3/prediction3.json'