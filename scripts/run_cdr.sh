export CUDA_VISIBLE_DEVICES=0

if true; then
bs=1
bl=3e-5
uls=(2e-5)
accum=1
seeds=(666 122 111 222 203)
for seed in ${seeds[@]}
  do
  for ul in ${uls[@]}
  do
  python3 -u  ./bio_train.py --data_dir ./dataset/cdr \
    --max_height 42 \
    --bert_lr $bl \
    --transformer_type bert \
    --model_name_or_path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --train_file train_filter.data \
    --dev_file dev_filter.data \
    --test_file test_filter.data \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 1 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 20 \
    --gnn_num_layer 4 \
    --seed $seed \
    --num_class 2 \
    --save_path checkpoints/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.pt \
    --log_dir logs/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.log
  done
done
fi
