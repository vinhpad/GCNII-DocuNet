export CUDA_VISIBLE_DEVICES=0

if true; then
type=context-based
bs=1
bl=3e-5
uls=(2e-5)
accum=1
seeds=(128 122 111 222 203)

for seed in ${seeds[@]}
  do
  for ul in ${uls[@]}
  do
  python3 -u  ./bio_train.py --data_dir ./dataset/gda \
    --max_height 42 \
    --channel_type $type \
    --bert_lr $bl \
    --transformer_type bert \
    --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --train_file train.data \
    --dev_file dev.data \
    --test_file test.data \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 1 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30 \
    --tau 0.7 \
    --gnn_num_layer 2 \
    --gnn_node_embedding 50 \
    --seed $seed \
    --num_class 2 \
    --save_path output/checkpoints/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.pt \
    --log_dir output/logs/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.log
  done
done
fi

# 87.81400467142865