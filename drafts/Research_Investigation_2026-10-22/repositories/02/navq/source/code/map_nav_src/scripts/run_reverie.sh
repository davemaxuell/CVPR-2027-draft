DATA_ROOT=../datasets

train_alg=dagger

features=clipbase
ft_dim=512
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed} #-${ngpus}gpus

outdir=${DATA_ROOT}/REVERIE/exprs_map/finetune/ORI${name}
Q_ckpt=/hy-tmp/OpenSource/code/datasets/Q-ckpt80000.pt
pretrain_ckpt=/hy-tmp/OpenSource/code/datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker/ckpts/model_step_80000.pt
Q_ckpt=/hy-tmp/OpenSource/code/datasets/Q-ckpt9000.pt
pretrain_ckpt=/hy-tmp/OpenSource/code/datasets/REVERIE/exprs_map/pretrain/ORIcmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker/ckpts/model_step_68000.pt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 4
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      --progress_bin 5 

      --WM_ckpt ${Q_ckpt}
      --use_P_loss 0.
      --feat_dropout_Q 0.
      "


# train
python reverie/main_nav_obj.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ${pretrain_ckpt} \
      --eval_first

# test
# python reverie/main_nav_obj.py $flag  \
#       --tokenizer bert \
#       --resume_file ../datasets/REVERIE/trained_models/best_val_unseen \
#       --test --submit