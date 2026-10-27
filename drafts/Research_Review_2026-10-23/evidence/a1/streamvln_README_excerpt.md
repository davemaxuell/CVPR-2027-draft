<br>
<p align="center">
<h1 align="center"><strong>StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling</strong></h1>
  <p align="center">
    <a href='https://github.com/kellyiss/' target='_blank'>Meng Wei*</a>&emsp;
    <a href='https://bryce-wan.github.io/' target='_blank'>Chenyang Wan*</a>&emsp;
    <a href='https://scholar.google.com/citations?user=CKWKIscAAAAJ&hl=en' target='_blank'>Xiqian Yu*</a>&emsp;
    <a href='https://tai-wang.github.io/' target='_blank'>Tai Wang*‡</a>&emsp;
    <a href='https://yuqiang-yang.github.io/' target='_blank'>Yuqiang Yang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=-zT1NKwAAAAJ&hl=en' target='_blank'>Xiaohan Mao</a>&emsp;
    <a href='https://zcmax.github.io/' target='_blank'>Chenming Zhu</a>&emsp;
    <a href='https://wzcai99.github.io/' target='_blank'>Wenzhe Cai</a>&emsp;
    <a href='https://hanqingwangai.github.io/' target='_blank'>Hanqing Wang</a>&emsp;
    <a href='https://yilunchen.com/about/' target='_blank'>Yilun Chen</a>&emsp;
    <a href='https://xh-liu.github.io/' target='_blank'>Xihui Liu†</a>&emsp;
    <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang†</a>&emsp;
    <br>
    Shanghai AI Laboratory&emsp;The University of Hong Kong&emsp;Zhejiang University&emsp;Shanghai Jiao Tong University&emsp;
  </p>
</p>

<div id="top" align="center">


[![arxiv](https://img.shields.io/badge/arXiv_2507.05240-red?logo=arxiv)](http://arxiv.org/abs/2507.05240)
[![project](https://img.shields.io/badge/Project_Page-0065D3?logo=rocket&logoColor=white)](https://streamvln.github.io/)
[![hf](https://img.shields.io/badge/Hugging_Face-FF9D00?logo=huggingface&logoColor=white)](https://huggingface.co/papers/2507.05240/)
[![video-en](https://img.shields.io/badge/YouTube-D33846?logo=youtube)](https://www.youtube.com/watch?v=gG3mpefOBjc)

</div>

## 🏠 About
<strong><em>StreamVLN</em></strong> generates action outputs from continuous video input in an online, multi-turn dialogue manner. Built on **LLaVA-Video** as the foundational Video-LLM, we extend it for interleaved vision, language, and action modeling. For both effective context modeling of long sequence and efficient computation for real-time interaction, StreamVLN has: (1) a **fast-streaming** dialogue context with a sliding-window KV cache; and (2) a **slow-updating** memory via token pruning.
<div style="text-align: center;">
    <img src="assets/teaser.gif" width=100% >
</div>

## 📢 News
[2025-09-28] We have updated the [checkpoint](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3) which is trained on R2R_VLNCE_v1-3, achieving better results: R2R (NE:4.90, OS:63.6, SR:56.4, SPL:50.2) and RxR (NE:5.65, SR:54.4, SPL:45.4, nDTW:63.7). **Please switch your training and testing data to R2R_VLNCE_v1-3 if you used R2R_VLNCE_v1 before.**

[2025-08-28] We have released the code and [guide](realworld/realworld.md) for real-world deployment on a unitree Go2 robot.

[2025-08-21] We have released the code for the following components: 1) **Dagger Data Collection**; 2) **Stage-Two Co-training** with the LLaVA-Video-178K, ScanQA, and MMC4 datasets.

[2025-07-30] We have released the ScaleVLN training data, including a subset of ~150k episodes converted from the discrete environment setting to the VLN-CE format. For usage details, see [here](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/README.md#envdrop--scalevln-dataset-note).

[2025-07-18] We’ve fixed a bug where num_history was not correctly passed to the model during evaluation, causing it to default to None. This had a significant impact on performance. Please make sure to pull the latest code for correct evaluation.

## 🛠 Getting Started
We test under the following environment:
* Python 3.9
* Pytorch 2.1.2
* CUDA Version 12.4 

1. **Preparing  a conda env with `Python3.9` & Install habitat-sim and habitat-lab**
    ```bash
    conda create -n streamvln python=3.9
    conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
    git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e habitat-lab  # install habitat_lab
    pip install -e habitat-baselines # install habitat_baselines
    ```

2. **Clone this repository**
    ```bash
    git clone https://github.com/OpenRobotLab/StreamVLN.git
    cd StreamVLN
    ```

<!-- 3. **Data Preparation**

    - You need to download the **Matterport3D (MP3D)** scenes first. Please follow the instructions in the [official project page](https://niessner.github.io/Matterport/). Place them in the `data/scene_datasets` folder.

    - For **evaluation**, please download the VLN-CE episodes: [r2r](https://dl.fbaipublicfiles.com/habitat/data/datasets/vln/mp3d/r2r/v1/vln_r2r_mp3d_v1.zip) and [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view), and place them in the `data/datasets` folder.

    - For **training**, please downlaod our observation-action pairs from [Hugging Face](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data), extract and place them in the `data/trajectory_data` folder. 

    The data folder should follow this structure:

    ```shell
    data/
    ├── datasets/
    │   ├── r2r
    │   │   ├── train/
    │   │   ├── val_seen/
    │   │   │   └── val_seen.json.gz
    │   │   └── val_unseen/
    │   │       └── val_unseen.json.gz
    │   └── rxr
    │       ├── train/
    │       ├── val_seen/
    │       │   ├── val_seen_guide.json.gz
    │       │   └── ...
    │       └── val_unseen/
    │           ├── val_unseen_guide.json.gz
    │           └── ...  
    ├── scene_datasets/
    │   └── mp3d/                   
    │       ├── 17DRP5sb8fy/        
    │       ├── 1LXtFkjw3qL/
    │       └── ...                 
    └── trajectory_data/
        ├── EnvDrop/
        │   ├── images/
        │   └── annotations.json
        ├── R2R/
        │   ├── images/
        │   └── annotations.json
        └── RxR/
            ├── images/
            └── annotations.json
    ``` -->

## 📁 Data Preparation

To get started, you need to prepare three types of data:

1. **Scene Datasets**  
   - For **R2R**, **RxR** and **EnvDrop**: Download the MP3D scenes from the [official project page](https://niessner.github.io/Matterport/), and place them under `data/scene_datasets/mp3d/`.
   - For **ScaleVLN**: Download the HM3D scenes from the [official github page](https://github.com/matterport/habitat-matterport-3dresearch), and place the `train` split under `data/scene_datasets/hm3d/`

2. **VLN-CE Episodes**  
   Download the VLN-CE episodes:
   - [r2r](https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view) (Rename `R2R_VLNCE_v1/` -> `r2r/`)
   - [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) (Rename `RxR_VLNCE_v0/` -> `rxr/`)
   - [envdrop](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) (Rename `R2R_VLNCE_v1-3_preprocessed/envdrop/` -> `envdrop/`)
   - [scalevln](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/ScaleVLN/scalevln_subset_150k.json.gz) (This is a subset of the ScaleVLN dataset, converted to the VLN-CE format. For the original dataset, please refer to the [official repository](https://github.com/wz0919/ScaleVLN).)
  
   Extract them into the `data/datasets/` directory.

3. **Collected Trajectory Data**  
  We provide pre-collected observation-action trajectory data for training. These trajectories were collected using the **training episodes** from **R2R** and **RxR** under the Matterport3D environment. For the **EnvDrop** and **ScaleVLN** subset, please refer to [here](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/README.md) for instructions on how to collect it yourself.
  Download the observation-action trajectory data from [Hugging Face](https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data), and extract it to `data/trajectory_data/`.

4. **Co-training Data Preparation**

    Download the respective datasets from their official sources and place them in the `data/co-training_data/`.

    - LLaVA-Video-178K: Available on Hugging Face at [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K).

    - ScanNet: 

      - The main dataset can be downloaded from the [official GitHub repository](https://github.com/ScanNet/ScanNet).

      - Download the annotation files `scanqa_annotations.json` and `sqa3d_annotations.json` from [here](https://huggingface.co/datasets/chchnii/StreamVLN-ScanQA-SQA3D-Data). These files are subsets of the [LLaVA-3D-DATA](https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Data).


    - MMC4-core: Available via the [official GitHub repository](https://github.com/allenai/mmc4).

Your final folder structure should look like this:

```bash
data/
├── datasets/
│   ├── r2r/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
│   ├── rxr/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   ├── val_seen_guide.json.gz
│   │   │   └── ...
│   │   └── val_unseen/
│   │       ├── val_unseen_guide.json.gz
│   │       └── ...
│   ├── envdrop/
│   │   ├── envdrop.json.gz
│   │   └── ...
│   └── scalevln/
│       └── scalevln_subset_150k.json.gz
├── scene_datasets/
│   └── hm3d/
│       ├── 00000-kfPV7w3FaU5/
│       ├── 00001-UVdNNRcVyV1/
│       └── ...
│   └── mp3d/
