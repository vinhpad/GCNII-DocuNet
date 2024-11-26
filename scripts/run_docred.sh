#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

# -------------------Training Shell Script--------------------
type=context-based
bs=4
bl=3e-5
ul=4e-4
accum=2
seeds=(3 5 7 11 13)
for seed in ${seeds[@]}
do
python -u ./docred_train.py --data_dir ./dataset/docred \
--bert_lr $bl \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size $bs \
--test_batch_size $bs \
--gradient_accumulation_steps $accum \
--num_labels 4 \
--learning_rate $ul \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30 \
--seed $seed \
--num_class 97 \
--save_path ./checkpoints/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}_seed_${seed}.pt \
--log_dir ./logs/docred/train_roberta-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}_seed_${seed}.log
done
