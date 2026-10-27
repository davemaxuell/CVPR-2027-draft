# Multi-Human Annotations
🚀🚀🚀 [**Download Here**](https://drive.google.com/drive/folders/1XvGHgLJ0MFDNY_k_iVwE_oGpfBfBaZif?usp=sharing)

# 🏃 Human Activities Annotation

## 🌍 **Stage 1: Coarse Annotation**
- **Goal:** Assign human motions to specific **regions** and **objects** using a **coarse-to-fine approach**.
- **Process:**
  - Filter human motions **$\mathbf{H}$** based on region **$\mathbf{R}$** and object list **$\mathbf{O}$**.
  - Match motions **$h_i$** with objects **$j_i$** using **semantic similarity**.
  - Optimize human placements **$\mathbf{p}_{opt}^{h_i}$** using **Particle Swarm Optimization (PSO)**.  
- **Constraints:**
  - Search space limited by **region boundaries**.
  - Maintain **minimum safe distance** $\epsilon = 1m$ from other objects.
  - Ensures **naturalistic human placements** for training navigation agents.

## 🎥 **Stage 2: Fine Annotation**
- **Inspired by:** Real-world **3D skeleton tracking** techniques.
- **Setup:**
  - **9 RGB cameras** surround each human model to refine **position & orientation**.
  - **Multi-view capture** to correct **clipping issues** with surrounding objects.
- **Camera Angles:**
  - **8 side cameras:** $\theta_{\text{lr}}^{i} = \frac{\pi i}{8}$, alternate **up/down tilt**.
  - **1 overhead camera:** $\theta_{\text{ud}}^{9} = \frac{\pi}{2}$.
- **Scale:** 529 human models annotated in **374 regions** across **90 scans**.

## 👥 **Multi-Human Interaction & Motion Enrichment**
- **Goal:** Increase **scene diversity** and **human interactions**.
- **Process:**
  - Use **LLMs** to generate new multi-human interactions.
  - **Manual refinement (4 rounds)** ensures consistency.
  - Place new motions relative to objects & use **multi-camera annotation**.
- **Result:**  
  - **910 human models** across **428 regions**.
  - **Complex motions**: Walking downstairs, climbing stairs.
  - **Interaction stats:** 72 **two-human pairs**, 59 **three-human pairs**, 15 **four-human groups**.
- **Impact:** Enables precise **social modeling** for human-aware navigation.

**We present several annotated instances of human subjects (Overall and single), showcasing a variety of well-aligned motions, movements, and interations.** 

<div align="center">
  <img src="../../demo/gifs/havln.gif" alt="image2" width="700"/>
</div>


**Overall View of Nine Annoated Scenarios from HA-VLN Simulator (90 scans in total)** 

<div align="center">
  <img src="../../demo/figs/overview_example-1.png" alt="image2" width="700"/>
</div>

**Single Humans with Movements (910 Humans in total)** 

Demo 1|Demo 2|Demo 3
--|--|--
<img src="../../demo/gifs/demo_1.gif" width="280">|<img src="../../demo/gifs/demo_2.gif" width="280">|<img src="../../demo/gifs/demo_3.gif" width="280">


Demo 4|Demo 5|Demo 6
--|--|--
<img src="../../demo/gifs/demo_4.gif" width="280">|<img src="../../demo/gifs/demo_5.gif" width="280">|<img src="../../demo/gifs/demo_6.gif" width="280">

