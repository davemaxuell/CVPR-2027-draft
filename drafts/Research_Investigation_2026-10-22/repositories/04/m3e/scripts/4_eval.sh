# Step 4: Evaluate model on all domains

CHECKPOINT= # The model from Step 3 momentum update (e.g., output/scene3/merged.pt)
PRETRAINED_MODEL=data/models/Vicuna-7B
PRECISION="amp_bf16"
EVAL_CFG_FILE=configs/r2r_all.yaml
VAL_SPLITS=(base_0 val_1 val_2 val_3 val_4 val_5 val_6 val_7 val_8)
VAL_BATCH=1
OUTPUT_DIR= # The output directory
MOE_ARGS="--moe_lora_r 16 --moe_lora_alpha 32 --moe_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

python train.py \
    --stage multi --mode test --data_dir data --cfg_file $EVAL_CFG_FILE \
    --pretrained_model_name_or_path $PRETRAINED_MODEL --precision $PRECISION \
    --resume_from_checkpoint $CHECKPOINT \
    --test_datasets R2R \
    --validation_split "${VAL_SPLITS[@]}" \
    --val_batch_size $VAL_BATCH --output_dir $OUTPUT_DIR \
    --save_pred_results --save_detail_results \
    $MOE_ARGS
