name=goat_r2r

DATA_ROOT=../datasets
train_alg=dagger
ft_dim=768
features=clip768
ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
backdoor_dict_file=${DATA_ROOT}/R2R/navigator/goat_r2r/logs/backdoor/backdoor_update_features.tsv
frontdoor_dict_file=${DATA_ROOT}/R2R/navigator/goat_r2r/logs/frontdoor/frontdoor_update_features.tsv
resume_file=${DATA_ROOT}/R2R/navigator/goat_r2r/ckpts/best_val_unseen.pt
vocabulary_path=${DATA_ROOT}/relevance_vocabularies/RxR_roberta_irrelevant_ids.txt

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --name ${name}   
      --mode valid

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 1
      --lr 2e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2  

      --feat_dropout 0
      --dropout 0.0

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door

      --do_front_txt
      --do_front_img
      --do_front_his

      --backdoor_dict_file ${backdoor_dict_file}
      --frontdoor_dict_file ${frontdoor_dict_file}

      --submit
      --test
      "

# train
python3 r2r/main_nav.py $flag \
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