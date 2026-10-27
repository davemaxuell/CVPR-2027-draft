DATA_ROOT=/data2/ljy/VLN-DUET/datasets

train_alg=grpo

features=clip.b16
ft_dim=512
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k-test

outdir=../datasets/R2R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 500
      --aug_times 9
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0."

# train
#CUDA_VISIBLE_DEVICES='8' python r2r/main_nav.py $flag  \
#      --tokenizer bert \
#      --bert_ckpt_file ../datasets/R2R/trained_models/pretrain/duet_vit-b16_model_step_140000.pt \
#      --resume_file ../datasets/R2R/trained_models/finetune/duet_vit-b16_ft_best_val_unseen \
#      --aug ../datasets/R2R/r2r_preprocess_data/annotations/R2R_scalevln_ft_aug_enc.json \

#CUDA_VISIBLE_DEVICES='8' python r2r/main_nav.py $flag  \
#      --tokenizer bert \
#      --bert_ckpt_file ../datasets/R2R/exprs_map/pretrain/cmt-clip.b16-mlm.sap-init.lxmert-aug.speaker-mlp/ckpts/model_step_200000.pt \
#      --aug ../datasets/R2R/r2r_preprocess_data/annotations/R2R_scalevln_ft_aug_enc.json \

# test
CUDA_VISIBLE_DEVICES='2' python r2r/main_nav.py $flag  \
      --tokenizer bert \
      --resume_file ../datasets/R2R/exprs_map/finetune/grpo-clip.b16-seed.0-init.aug.45k-scale-dr.grpo-16/ckpts/best_val_unseen \
      --test --submit