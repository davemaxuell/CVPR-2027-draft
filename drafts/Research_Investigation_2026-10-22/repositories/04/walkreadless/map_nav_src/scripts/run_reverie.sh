name=goat_reverie
DATA_ROOT=../datasets

train_alg=dagger
features=clip768
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
ngpus=1
seed=0

outdir=${DATA_ROOT}/REVERIE/
backdoor_dict_file=${DATA_ROOT}/REVERIE/navigator/goat_reverie/logs/backdoor/backdoor_update_features.tsv
frontdoor_dict_file=${DATA_ROOT}/REVERIE/navigator/goat_reverie/logs/frontdoor/frontdoor_update_features.tsv
resume_file=${DATA_ROOT}/REVERIE/navigator/goat_reverie/ckpts/best_val_unseen.pt
vocabulary_path=${DATA_ROOT}/relevance_vocabularies/RxR_roberta_irrelevant_ids.txt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --mode valid
      --name ${name}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 80
      --max_objects 20

      --batch_size 1
      --lr 2e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0
      --dropout 0

      --do_back_img
      --do_back_txt
      --z_instr_update
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door

      --do_front_img
      --do_front_his
      --do_front_txt

      --backdoor_dict_file ${backdoor_dict_file}
      --frontdoor_dict_file ${frontdoor_dict_file}

      --submit
      "
# train
python3 -u reverie/main_nav_obj.py $flag  \
      --resume_file ${resume_file} \
      --vocabulary_path ${vocabulary_path} \
      --vp_crop \
      --vp_crop_method attn_scores \
      --crop_vp_len_pano 0 \
      --gmap_pruning_t_threshold 0 \
      --txt_crop \
      --txt_pruning_rate 0.0 \
      --txt_prune_strat vocabulary \
      # --val_unseen \