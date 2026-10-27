name=goat_rxr_setup_1
DATA_ROOT=../datasets

train_alg=dagger
ft_dim=768
features=clip768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/RxR/
backdoor_dict_file=${DATA_ROOT}/RxR/navigator/goat_rxr/logs/backdoor/backdoor_update_features.tsv
frontdoor_dict_file=${DATA_ROOT}/RxR/navigator/goat_rxr/logs/frontdoor/frontdoor_update_features.tsv
resume_file=${DATA_ROOT}/RxR/navigator/goat_rxr/ckpts/best_val_unseen.pt
vocabulary_path=${DATA_ROOT}/relevance_vocabularies/RxR_roberta_irrelevant_ids.txt

flag="--root_dir ${DATA_ROOT}
      --dataset rxr
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta   
      --name ${name}   
      --mode valid

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy ndtw
      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 28
      --max_instr_len 250

      --batch_size 1

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0
      --dropout 0

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door

      --do_front_img
      --do_front_his
      --do_front_txt

      --frontdoor_dict_file ${frontdoor_dict_file}
      --backdoor_dict_file ${backdoor_dict_file}

      "

# train
python3 r2r/main_nav.py $flag  \
      --resume_file ${resume_file} \
      --vocabulary_path ${vocabulary_path} \
      --vp_crop \
      --vp_crop_method tome \
      --crop_vp_len_pano 5 \
      --gmap_pruning_t_threshold 8 \
      --txt_crop \
      --txt_pruning_rate 0.2 \
      --txt_prune_strat vocabulary \
      --val_unseen \
      
      
      # --vp_crop_method medium_2 \
      # --vp_crop_dropout 0.0 \

      # --saliency_debug \
      # --test_flop \

      

      

      

      