export CUDA_VISIBLE_DEVICES=0

if true; then
type=context-based
bs=4
bl=3e-5
uls=(3e-4)
accum=1
seeds=(128)

for seed in ${seeds[@]}
  do
  for ul in ${uls[@]}
  do
  python3 grace_train.py \
    --data_dir dataset/cdr \
    --transformer_type bert \
    --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --train_file train_filter.data \
    --dev_file dev_filter.data \
    --test_file test_filter.data \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 2 \
    --learning_rate $ul \
    --bert_lr $bl \
    --transformer_type bert \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 1 \
    --gnn_num_layer 2 \
    --gnn_node_type_embedding 128 \
    --gnn_hidden_feat_dim 512 \
    --gnn_output_dim 256 \
    --grace_projection_hidden_feat_dim 512 \
    --grace_projection_out_feat_dim 256 \
    --seed $seed \
    --num_class 2 \
    --save_path checkpoints/cdr/train_grace_lr${ul}_accum${accum}_bs${bs}.pt \
    --log_dir logs/cdr/train_grace_lr${lr}_accum${accum}_bs${bs}.log \
    --grace_loss_viz logs/cdr/train_grace_loss_viz_lr${lr}_accum${accum}_bs${bs}.png \
    --feature_prob_first 0.1 \
    --feature_prob_second 0.2 \
    --edge_prob_first 0.2 \
    --edge_prob_second 0.1 \
    --tau 0.7 
  done
done
fi
