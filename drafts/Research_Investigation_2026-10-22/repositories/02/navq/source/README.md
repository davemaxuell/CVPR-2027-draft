# Implementation of "NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation" 

This is the PyTorch implementation of ["NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation"](https://www.arxiv.org/abs/2510.16457) (ICCV 2025)

# Installation

Please follow [DUET](https://github.com/cshizhe/VLN-DUET) to prepare the Matterport3D simulator, the environment, and the data and features of REVERIE and R2R.

After that, please download the following files and put them under the `datasets` drectory: 
1. The [CLIP features of the image observations](https://huggingface.co/datasets/OpenGVLab/ScaleVLN/blob/main/features.zip), provided by [ScaleVLN](https://github.com/wz0919/ScaleVLN) (`clip_vit-b16_mp3d_original.hdf5`, `clip_vit-b16_mp3d_hm3d_gibson.hdf5`)
2. The [CLIP features of the textual descriptions](https://huggingface.co/datasets/woyut/NavQ_ICCV25-files) (`text_fts_LangNavGlobal.npy`, `text_fts_LangNavGlobal_SVLN.npy`). The textual descriptions on MP3D are borrowed from [LangNav](https://github.com/pbw-Berwin/LangNav), while the textual descriptions on the new scenes of ScaleVLN are extracted using [BLIP](https://github.com/salesforce/BLIP) in a similar mannar
3. The `bert-base-uncased` [model](https://huggingface.co/google-bert/bert-base-uncased)

Also, please download [these files](https://huggingface.co/datasets/woyut/NavQ_ICCV25-files/tree/main/Q_files) related to the Q-Model training and put them under the `Q_pretrain/Q_files` directory. 

# Training the Q Model

To train the Q Model, you may first get in `Q_pretrain` and use the following codes.

```
python train_mae.py --out_dir=ckpt/MAE --use_SVLN=False --eval_interval=1000 
python train.py --out_dir=ckpt/Q --use_SVLN=False --eval_interval=1000 --resume_ckpt=ckpt/MAE/ckpt20000.pt
```

To use the additionl scenes provided by [ScaleVLN](https://github.com/wz0919/ScaleVLN), please run:

```
python train_mae.py --out_dir=ckpt/MAE-SVLN --use_SVLN=True --eval_interval=2000 
python train.py --out_dir=ckpt/Q-SVLN --use_SVLN=True --eval_interval=2000 --resume_ckpt=ckpt/MAE-SVLN/ckpt100000.pt
```

Our Pre-trained Q-Model can be found at [here](https://huggingface.co/datasets/woyut/NavQ_ICCV25-files/tree/main/CKPT_release).

# Agent Pretraining

Please modify the `"WM_ckpt"` in `pretrain_src/config/reverie_obj_model_config.json` to the path of the Q-model you would like to use.

Then, run the following lines to start pretraining the agent.

```
cd pretrain_src
bash run_reverie.sh
```

# Agent Finetuning

Please modify the `Q_ckpt` and `pretrain_ckpt` in `map_nav_src/scripts/run_reverie.sh` to the path of the Q-model and the pretrained model.

Then, run the following lines to start finetuning the agent.

```
cd map_nav_src
bash scripts/run_reverie.sh
```

We have provided our trained model at [here](https://huggingface.co/datasets/woyut/NavQ_ICCV25-files/tree/main/CKPT_release).

# Acknowledgement

This repository is built upon [DUET](https://github.com/cshizhe/VLN-DUET). The structure and training of the Q model are largely inspired by [nanoGPT](https://github.com/karpathy/nanoGPT). We also make use of the data provided by [ScaleVLN](https://github.com/wz0919/ScaleVLN) and [LangNav](https://github.com/pbw-Berwin/LangNav). We sincerely thank these works for their valuable contributions.
