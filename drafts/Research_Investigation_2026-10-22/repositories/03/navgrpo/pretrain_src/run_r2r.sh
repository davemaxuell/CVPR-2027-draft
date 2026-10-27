
NODE_RANK=0
NUM_GPUS=1
outdir=../datasets/R2R/exprs_map/pretrain/cmt-clip.b16-mlm.sap-init.lxmert-aug.speaker-mlp

# train
CUDA_VISIBLE_DEVICES='8' python train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_scalevln_pretrain.json \
    --output_dir $outdir
