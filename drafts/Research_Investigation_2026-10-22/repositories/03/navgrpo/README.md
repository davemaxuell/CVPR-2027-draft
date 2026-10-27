<div align="center">

# 🧭 NavGRPO

### Trajectory-Diversity-Driven Robust Vision-and-Language Navigation

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Trajectory-Diversity-Driven_Robust_Vision-and-Language_Navigation_CVPRF_2026_paper.html">
    <img src="https://img.shields.io/badge/arXiv-2506.xxxxx-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
  <a href="https://cocoastar.github.io/NavGRPO">
    <img src="https://img.shields.io/badge/Project-Page-1a73e8.svg?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://openaccess.thecvf.com/">
    <img src="https://img.shields.io/badge/CVPR-2026-6f42c1.svg?style=for-the-badge" alt="CVPR 2026"></a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="#-news">News</a> &nbsp;•&nbsp;
  <a href="#-overview">Overview</a> &nbsp;•&nbsp;
  <a href="#-method">Method</a> &nbsp;•&nbsp;
  <a href="#-installation">Installation</a> &nbsp;•&nbsp;
  <a href="#-usage">Usage</a> &nbsp;•&nbsp;
  <a href="#-results">Results</a> &nbsp;•&nbsp;
  <a href="#-citation">Citation</a>
</p>

**[Jiangyang Li](#)<sup>1</sup>, [Cong Wan](#)<sup>4</sup>, [Songlin Dong](#)<sup>2,3&dagger;</sup>, [Chenhao Ding](#)<sup>4</sup>, [Qiang Wang](#)<sup>1</sup>, [Zhiheng Ma](#)<sup>2,3</sup>, [Yihong Gong](#)<sup>1</sup>**

<sup>1</sup>Xi'an Jiaotong University &nbsp;&nbsp; <sup>2</sup>Shenzhen University of Advanced Technology<br>
<sup>3</sup>Guangdong Provincial Key Laboratory of Computility Microelectronics &nbsp;&nbsp; <sup>4</sup>School of Software Engineering, XJTU

<sup>&dagger;</sup>Corresponding author

<br>

<!-- TEASER PLACEHOLDER -->
<img src="assets/head.png" width="95%" alt="Head">

<p align="left"><sub><b>Figure 1.</b> Navigation behavior under early-stage perturbations in unfamiliar environments. The
baseline IL agent struggles to recover from errors due to limited exposure to failed trajectories, often
detouring or failing to reach the goal. Our NavGRPO agent learns from diverse rollouts, enabling robust
error correction and successful navigation despite perturbations.</sub></p>

</div>

---

## 📰 News

- **`2026-3-22`** 🎉 NavGRPO is accepted to **CVPR 2026**!
- **`2026-7-9`** 🚀 Code and pretrained checkpoints released.
- **`2026-6-6`** 📄 Paper available on [OpenACCESS](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Trajectory-Diversity-Driven_Robust_Vision-and-Language_Navigation_CVPRF_2026_paper.html).

---

## ✅ TODO

- [x] Release training & evaluation code
- [x] Release configs for R2R / REVERIE / R2R-CE
- [ ] Release pretrained checkpoints (ScaleVLN-NavGRPO, DUET-NavGRPO)
- [ ] Release perturbation-robustness evaluation scripts
- [ ] Add Colab / demo notebook

---

## 📖 Overview

> **TL;DR** — NavGRPO replaces imitation-learning objectives with **Group Relative Policy Optimization**,
> learning from *diverse* rollouts (both successes and failures) rather than copying expert paths. This
> gives large gains in **generalization** and, especially, **robustness to perturbations**.

Vision-and-Language Navigation (VLN) requires agents to navigate photo-realistic environments following
natural language instructions. Current methods rely predominantly on **imitation learning (IL)**, which
imitates expert demonstrations and therefore suffers from two fundamental limitations: **(i)** limited
generalization to unseen environments, and **(ii)** poor robustness once the agent deviates from the
expert path and enters unseen state distributions with no recovery policy.

**NavGRPO** is a reinforcement learning framework that learns goal-directed navigation policies through
**GRPO**. By exploring diverse trajectories and optimizing via within-group performance comparisons, the
agent distinguishes effective strategies *beyond* expert paths — without any additional value network.

<table>
<tr><td>

**Why it works**
- 🎯 **No value network** — advantages from within-group reward ranking (Dr.GRPO), avoiding value-net instability in VLN's huge action space.
- 📐 **Geometric reward** — trajectory-level success + path efficiency, modulated by a step-level progress coefficient.
- ♻️ **Hard-case replay** — supervised refinement on persistently failing instructions prevents forgetting at near-zero extra cost.
- 🔌 **Plug-and-play** — validated on **ScaleVLN** and **DUET** under one unified interface.

</td></tr>
</table>

---

## 🧠 Method

<div align="center">
<!-- FRAMEWORK FIGURE PLACEHOLDER -->
<img src="assets/method.png" width="95%" alt="NavGRPO framework">
<p align="left"><sub><b>Figure 2.</b> Overview of the NavGRPO training framework. For each instruction we sample K diverse
trajectories through policy rollout, compute rewards using trajectory- and step-level signals, estimate
group-relative advantages by comparing within instruction groups, and optimize the policy through debiased
advantage estimation without value networks.</sub></p>
</div>

NavGRPO is built on three design choices (paper §3):
<details>
<summary><b>📊 Click to expand full equation set</b></summary>

<br>

| Component | Equation |
|:--|:--|
| Policy | $\pi_\theta(a_t \mid \mathcal{W}, \mathcal{O}_t, \mathcal{M}_t) = \operatorname{softmax}(f_\theta(\mathcal{W}, \mathcal{O}_t, \mathcal{M}_t))$ |
| Debiased advantage | $\hat{A}_k = r_k - \operatorname{mean}(\\{r_1,\dots,r_K\\})$ |
| Total reward | $r_k = R_{\text{nav}}(d_k) + \alpha \cdot R_{\text{path}}(L_k, L^\*)$ |
| Navigation success | $R_{\text{nav}}(d_k) = \exp(-d_k^2 / 2\epsilon^2)\cdot \mathbb{1}(d_k < \epsilon)$ |
| Path efficiency | $R_{\text{path}}(L_k, L^\*) = -\max(L_k - L^\*, 0)/L^\*$ |
| Step-progress coeff. | $\gamma_{k,t} = 1 + \operatorname{sign}(\hat{A}_k)\cdot (d_{t-1}-d_t)/L^\*$ |
| GRPO objective | $\mathcal{J} = \mathbb{E}\big[\tfrac{1}{K}\sum_k \sum_t (\mathcal{J}_{\text{clip}}(k,t) - \beta\, \mathbb{D}_{\text{KL}}[\pi_\theta \\| \pi_{\text{ref}}])\big]$ |

</details>

---

## 🔧 Installation

```bash
# 1 · Clone
git clone https://github.com/cocoastar/NavGRPO.git
cd NavGRPO

# 2 · Environment
conda create -n navgrpo python=3.9 -y
conda activate navgrpo

# 3 · PyTorch (match your CUDA)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 4 · NavGRPO + dependencies
pip install -r requirements.txt
pip install -e .
```

> 💡 For the Matterport3D simulator (R2R / REVERIE) and Habitat (R2R-CE) setup, see
> [`docs/INSTALL.md`](docs/INSTALL.md).

---

## 📦 Data Preparation

NavGRPO uses **R2R**, **REVERIE**, and **R2R-CE**, following the standard
[DUET](https://github.com/cshizhe/VLN-DUET) / [ScaleVLN](https://github.com/wz0919/ScaleVLN) layout.
Full download links and preprocessing steps are in [`docs/DATA.md`](docs/DATA.md).

```
data/
├── connectivity/               # navigation graphs (*.json)
├── R2R/
│   ├── annotations/            # {train,val_seen,val_unseen}_enc.json
│   └── features/               # image features (*.hdf5)
├── REVERIE/
│   ├── annotations/
│   └── features/
└── R2R-CE/                     # Habitat / waypoint-predictor assets
```

---

## 🚀 Usage

### Training

```bash
# ScaleVLN-NavGRPO — supervised warm-up (30k) then GRPO fine-tuning (200k total)
bash scripts/train_r2r.sh
bash scripts/train_reverie.sh
bash scripts/train_r2r_ce.sh
```

Or launch the trainer directly with a config:

```bash
python -m navgrpo.train \
    --config configs/scalevln_navgrpo_r2r.yaml \
    --output_dir outputs/r2r_scalevln_navgrpo \
    --seed 0
```

### Evaluation

```bash
bash scripts/eval_r2r.sh <ckpt_path> val_unseen
```

### Robustness experiments (paper Table 1)

```bash
# Global perturbation — sample from policy with prob p at each step
python -m navgrpo.perturb --resume <ckpt> --mode global --prob 0.4

# Early perturbation — least-probable action for the first N steps
python -m navgrpo.perturb --resume <ckpt> --mode early --steps 3
```

---

## 📊 Results

### Comparison with state-of-the-art on R2R & REVERIE

<sup>†: RL-based methods. Best in **bold**.</sup>

| Method | R2R Val-U · NE↓ | SR↑ | SPL↑ | R2R Test-U · NE↓ | SR↑ | SPL↑ | REVERIE Val-U · OSR↑ | SR↑ | SPL↑ | REVERIE Test-U · OSR↑ | SR↑ | SPL↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| RCM<sup>†</sup>          | 5.88 | 43 | –  | 6.12 | 43 | 38 | 14.23 | 9.29  | 6.97  | 11.68 | 7.84  | 6.67  |
| SERL<sup>†</sup>         | 4.74 | 56 | 48 | 5.63 | 53 | 49 | –     | –     | –     | –     | –     | –     |
| VLN&#8635;BERT       | 3.93 | 63 | 57 | 4.09 | 63 | 57 | 35.02 | 30.67 | 24.90 | 32.91 | 29.61 | 23.99 |
| HAMT<sup>†</sup>         | 3.65 | 66 | 61 | 3.93 | 65 | 60 | 36.84 | 32.95 | 30.20 | 33.41 | 30.40 | 26.67 |
| HOP+                 | 3.49 | 67 | 61 | 3.71 | 66 | 60 | 40.04 | 36.07 | 31.13 | 35.81 | 33.82 | 28.24 |
| BEVBert              | 2.81 | 75 | 64 | 3.13 | 73 | 62 | 56.40 | 51.78 | 36.37 | 57.26 | 52.81 | 36.41 |
| KERM                 | 3.22 | 72 | 61 | 3.61 | 70 | 59 | 55.21 | 50.44 | 35.38 | 57.58 | 52.43 | 39.21 |
| GridMM               | 2.83 | 75 | 64 | 3.35 | 73 | 62 | 57.48 | 51.37 | 36.47 | 59.55 | 53.13 | 36.60 |
| VER                  | 2.80 | 76 | 65 | 2.74 | 76 | 66 | 61.09 | 55.98 | 39.66 | 62.22 | 56.82 | 38.76 |
| GOAT                 | 2.40 | 78 | 68 | 3.04 | 75 | 65 | –     | 53.37 | 36.70 | –     | 57.72 | 40.53 |
| NavQ                 | 3.06 | 73 | 63 | 3.30 | 72 | 63 | 60.47 | 53.22 | 38.89 | 60.39 | 53.29 | 39.50 |
| DUET                 | 3.31 | 72 | 60 | 3.65 | 69 | 59 | 51.07 | 46.98 | 33.73 | 56.91 | 52.51 | 36.06 |
| **DUET-NavGRPO**     | 3.18 | 74 | 63 | 3.39 | 71 | 62 | 53.17 | 49.47 | 35.32 | 58.32 | 54.49 | 38.16 |
| ScaleVLN             | 2.34 | 79 | 70 | 2.73 | 77 | 68 | 63.85 | 56.97 | 41.84 | 62.65 | 56.13 | 39.52 |
| **ScaleVLN-NavGRPO** | **2.19** | **82** | **73** | **2.52** | **79** | **70** | **65.19** | **58.91** | **43.55** | **64.21** | **58.25** | **41.34** |

### R2R-CE (continuous environments)

<sup>†: methods using a candidate waypoint predictor for high-level action space.</sup>

| Method | Val-U · NE↓ | SR↑ | SPL↑ | Test-U · NE↓ | SR↑ | SPL↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| ETPNav<sup>†</sup>                | 4.71 | 57 | 49 | 5.12 | 55 | 48 |
| DUET<sup>†</sup>                 | 5.26 | 47 | 39 | 5.82 | 42 | 36 |
| **DUET-NavGRPO<sup>†</sup>**         | 5.02 | 50 | 42 | 5.64 | 44 | 38 |
| ScaleVLN<sup>†</sup>             | 4.80 | 55 | 51 | 5.11 | 55 | 50 |
| **ScaleVLN-NavGRPO<sup>†</sup>**     | **4.69** | **57** | **53** | **5.01** | **57** | **52** |

### Robustness to perturbations (R2R Val-U)

ΔSPL = SPL degradation from the unperturbed setting (smaller magnitude ⇒ more robust).

| Setting | Method | OSR↑ | NE↓ | SR↑ | SPL↑ | ΔSPL |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| **Global** · p=0.4 | ScaleVLN | 87.19 | 2.42 | 78.67 | 65.90 | −4.07 |
|                    | **NavGRPO**  | **88.79** | **2.22** | **81.06** | **71.69** | **−0.96** |
| **Global** · p=0.8 | ScaleVLN | 87.06 | 2.52 | 77.61 | 61.94 | −8.03 |
|                    | **NavGRPO**  | **88.49** | **2.27** | **79.95** | **69.98** | **−2.67** |
| **Early** · N=3    | ScaleVLN | 83.52 | 3.04 | 74.16 | 47.77 | −22.20 |
|                    | **NavGRPO**  | **82.96** | **2.82** | **75.35** | **62.66** | **−9.99** |

> 📌 At global perturbation p=0.4, NavGRPO drops only **0.96 SPL** vs. the baseline's 4.07 (**4.2× more
> robust**). At p=0.8 the advantage widens to **3.0×**, and NavGRPO's *perturbed* SPL (69.98) even exceeds
> the baseline's *unperturbed* score (69.97).

### Ablations (R2R Val-U)

<table>
<tr>
<td valign="top">

**(a) Reward components**

| nav | path | step | SPL↑ |
|:--:|:--:|:--:|:--:|
| ✓ |   |   | 71.07 |
| ✓ | ✓ |   | 71.88 |
| ✓ |   | ✓ | 71.93 |
| ✓ | ✓ | ✓ | **72.65** |

</td>
<td valign="top">

**(b) Group-opt variant**

| Method | SPL↑ |
|:--|:--:|
| w/o Group | 69.97 |
| GRPO | 71.23 |
| GSPO | 71.68 |
| GMPO | 71.62 |
| **Dr.GRPO** | **72.65** |

</td>
<td valign="top">

**(c) Trajectory num. K**

| K | SPL↑ |
|:--:|:--:|
| 2 | 71.02 |
| 4 | 71.98 |
| **8** | **72.65** |
| 16 | 72.87 |

</td>
</tr>
</table>

**(d) Alternative RL methods & training strategy**

| RL method | SR↑ | SPL↑ | &nbsp; | Training strategy | SR↑ | SPL↑ |
|:--|:--:|:--:|:--:|:--|:--:|:--:|
| SFT only        | 79.40 | 69.97 | &nbsp; | SFT only          | 79.40 | 69.97 |
| SFT+REINFORCE   | 79.28 | 69.85 | &nbsp; | RL only           | 64.30 | 59.82 |
| SFT+A2C         | 79.65 | 70.25 | &nbsp; | Sequential SFT-RL | 81.25 | 72.08 |
| SFT+PPO         | 79.95 | 70.68 | &nbsp; | **Ours** (+ replay) | **81.88** | **72.65** |
| **SFT+GRPO**    | **81.88** | **72.65** | &nbsp; | | | |

<details>
<summary><b>⚙️ Default hyperparameters</b></summary>

<br>

| Trajectories K | Batch B | Clip δ | KL β | Path α | Buffer M | LR | Warm-up | Total steps |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 8 | 8 | 0.2 | 0.01 | 0.25 | 200 | 1e-5 | 30k | 200k |

</details>

---

## 🖼️ Qualitative Results

<div align="center">
<!-- QUALITATIVE FIGURE PLACEHOLDER -->
<img src="assets/result.png" width="95%" alt="Qualitative comparison">
<p align="left"><sub><b>Figure 3.</b> Qualitative comparison on challenging instructions under normal conditions (top) and
initial perturbations (bottom). ScaleVLN fails to recover from errors in both scenarios, while NavGRPO
completes the tasks and demonstrates robust error correction under perturbations.</sub></p>
</div>

---

## 📜 Citation

If you find NavGRPO useful for your research, please consider giving a ⭐ and citing:

```bibtex
@inproceedings{li2026navgrpo,
  title     = {Trajectory-Diversity-Driven Robust Vision-and-Language Navigation},
  author    = {Li, Jiangyang and Wan, Cong and Dong, Songlin and Ding, Chenhao
               and Wang, Qiang and Ma, Zhiheng and Gong, Yihong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

This codebase builds upon the excellent open-source work of
[DUET](https://github.com/cshizhe/VLN-DUET),
[ScaleVLN](https://github.com/wz0919/ScaleVLN),
[HAMT](https://github.com/cshizhe/VLN-HAMT), and the
[Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator).
We thank the authors for releasing their code and data.

This work was supported by the National Natural Science Foundation of China (No. U21B2048,
No. 62576224), the Shenzhen Key Technical Projects, and the Guangdong Provincial Key Laboratory of
Computility Microelectronics (No. 2024B1212010007).

---

## 📄 License

This project is released under the [MIT License](LICENSE).

<div align="center">
<sub>⭐ Star us on GitHub if this project helps you!</sub>
</div>
