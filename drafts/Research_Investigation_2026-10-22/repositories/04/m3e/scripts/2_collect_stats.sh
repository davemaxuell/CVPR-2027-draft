# Step 2: Collect routing statistics and compute importance scores

CFG_FILE=configs/r2r_scanID.yaml
PRETRAINED_MODEL=data/models/Vicuna-7B
PRECISION="amp_bf16"
CHECKPOINT= # The model from Step 1 training (e.g., output/scene2/epoch_*.pt(for scene2)
STATS_DIR= # The output directory
MOE_ARGS="--moe_lora_r 16 --moe_lora_alpha 32 --moe_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

python tools/momentum/experts_stats.py \
    --stage multi --mode test --data_dir data \
    --cfg_file $CFG_FILE \
    --pretrained_model_name_or_path $PRETRAINED_MODEL --precision $PRECISION \
    --resume_from_checkpoint $CHECKPOINT \ The model from Step 1 training
    --test_datasets R2R \
    --validation_split train \
    --val_batch_size 1 \
    --output_dir $STATS_DIR \
    --importance_output "$STATS_DIR/importance.json" \
    $MOE_ARGS
