#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ./docred_train.py --data_dir ./dataset/docred \
--load_path \
--bert_lr 3e-5 \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 2 \
--num_labels 4 \
--learning_rate 4e-4 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30 \
--seed 5 \
--num_class 97 \
--save_path ./checkpoint/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}_seed_${seed}.pt \
--log_dir ./logs/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}_seed_${seed}.log
done
