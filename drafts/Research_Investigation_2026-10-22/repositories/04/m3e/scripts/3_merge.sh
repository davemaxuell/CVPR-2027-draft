# Step 3: Fine-grained momentum update

OLD_CKPT= # The merged model from the previous scene's continual learning (e.g., output/scene2/merged.pt)
NEW_CKPT= # The model from Step 1 training (e.g., output/scene3/epoch_*.pt(for scene3))
IMPORTANCE= # The importance scores from Step 2 (e.g., output/scene3/stats/importance.json)
OUTPUT= # The output directory

python tools/momentum/momentum_update.py \
    --old_ckpt $OLD_CKPT \
    --new_ckpt $NEW_CKPT \
    --importance $IMPORTANCE \
    --output $OUTPUT
