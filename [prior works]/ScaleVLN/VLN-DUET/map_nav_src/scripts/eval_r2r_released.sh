#!/bin/bash
# Evaluate a released ScaleVLN R2R checkpoint on val_train_seen / val_seen / val_unseen
# and write test-split predictions.
# usage: bash scripts/eval_r2r_released.sh {b16|h14} [gpu]    (gpu: physical index or MIG UUID)
set -e
cd "$(dirname "$0")/.."    # map_nav_src

if [ "${CONDA_DEFAULT_ENV:-}" != "scalevln" ]; then
    source /home/bufsgpu/yes/etc/profile.d/conda.sh
    conda activate scalevln
fi
export PYTHONPATH="$(cd ../.. && pwd)/Matterport3DSimulator/build:${PYTHONPATH:-}"
if [ -n "${2:-}" ]; then export CUDA_VISIBLE_DEVICES=$2; fi

DATA_ROOT=../datasets
model=${1:?"usage: eval_r2r_released.sh b16|h14 [gpu]"}

case $model in
    b16) features=clip.b16; ft_dim=512;  bs=16 ;;
    h14) features=clip.h14; ft_dim=1024; bs=8 ;;   # --env_aug only changes training features
    *) echo "unknown model: $model"; exit 1 ;;
esac

ckpt=${DATA_ROOT}/R2R/trained_models/finetune/duet_vit-${model}_ft_best_val_unseen
outdir=${DATA_ROOT}/R2R/exprs_map/eval/released_${model}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size 1
      --seed 0
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg dagger

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 15
      --max_instr_len 200

      --batch_size ${bs}

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --feat_dropout 0.4
      --dropout 0.5"

python r2r/main_nav.py $flag \
      --resume_file ${ckpt} \
      --test --submit --detailed_output

echo
cat ${outdir}/logs/valid.txt | grep "Env name"
