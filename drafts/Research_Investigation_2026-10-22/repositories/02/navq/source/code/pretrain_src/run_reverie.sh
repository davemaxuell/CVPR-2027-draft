NODE_RANK=0
NUM_GPUS=1
outdir=../datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker
outdir=../datasets/REVERIE/exprs_map/pretrain/ORIcmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker

# train
python train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_obj_pretrain.json \
    --output_dir $outdir --train_batch_size 16 --gradient_accumulation_steps 2 --n_workers 1 

