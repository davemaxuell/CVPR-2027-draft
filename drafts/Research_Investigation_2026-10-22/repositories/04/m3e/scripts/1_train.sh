# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# activate environment
# conda activate M3E

# Step 1: Train on a new scene
CFG_FILE=configs/r2r_scanID.yaml
PRETRAINED_MODEL=data/models/Vicuna-7B
PRECISION="amp_bf16"
RESUME_CKPT= # The merged model from the previous scene's continual learning (e.g., output/scene1/merged.pt(for scene2) or data/model_with_pretrain.pt(for scene1))
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEP=1
LEARNING_RATE=2e-5
NUM_EPOCHS=5
SEED=0
OUTPUT_DIR= # The output directory

MOE_ARGS="--moe_lora_r 16 --moe_lora_alpha 32 --moe_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

python train.py \
    --stage multi --cfg_file $CFG_FILE \
    --data_dir data --pretrained_model_name_or_path $PRETRAINED_MODEL --precision $PRECISION \
    --resume_from_checkpoint $RESUME_CKPT \
    --batch_size $BATCH_SIZE --gradient_accumulation_step $GRADIENT_ACCUMULATION_STEP --lr $LEARNING_RATE --seed $SEED --num_epochs $NUM_EPOCHS \
    --test_datasets R2R --validation_split val_unseen \
    --max_saved_checkpoints 1 --output_dir $OUTPUT_DIR \
    $MOE_ARGS
