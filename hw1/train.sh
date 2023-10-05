python my-multiple-choice.py  \
--model_name_or_path="bert-base-chinese"\
--cache_dir="./cache" \
\
--train_file="./data/smaller_train.json" \
--validation_file="./data/smaller_valid.json" \
--context_file="./data/smaller_context.json" \
--test_file="./data/test_data"\
--preprocessing_num_workers=6 \
\
--output_dir="./trained" \
--do_train \
--do_eval \
--num_train_epochs=9 \
--auto_find_batch_size \
--gradient_accumulation_steps=4 \
--learning_rate=2e-5 \
--warmup_steps=300 \
--dataloader_num_workers=6 \
\
--evaluation_strategy="steps" \
--eval_steps=500 \
--save_steps=500 \
--metric_for_best_model="accuracy" \
--load_best_model_at_end  \
--report_to="tensorboard" \
\
--bf16 --tf32=y \

