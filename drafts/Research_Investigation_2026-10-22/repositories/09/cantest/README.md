<div align="center" style="font-family: charter;">

 <h1>Where Did It Go Wrong? Capability-Oriented Failure Attribution for Vision-and-Language Navigation Agents</h1>
 
<img src="images/overview.png" width="90%"/>
<br />


<br />
<p align="justify">
Embodied agents in safety-critical applications such as Vision-Language Navigation (VLN) rely on multiple interdependent capabilities (e.g., perception, memory, planning, decision), making failures difficult to localize and attribute. Existing testing methods are largely system-level and provide limited insight into which capability deficiencies cause task failures. We propose a capability-oriented testing approach that enables failure detection and attribution by combining (1) adaptive test case generation via seed selection and mutation, (2) capability oracles for identifying capability-specific errors, and (3) a feedback mechanism that attributes failures to capabilities and guides further test generation. Experiments show that our method discovers more failure cases and more accurately pinpoints capability-level deficiencies than state-of-the-art baselines, providing more interpretable and actionable guidance for improving embodied agents.

</p>
</div>

# Preparation

## Environment

This project is developed with Python 3.9. You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/) to create the environment:

```bash
conda create -n lhvln python=3.9
conda activate lhvln
```

We use [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/main) as simulator, which can be [built from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source) or installed from conda:

```bash
conda install habitat-sim==0.3.1 headless -c conda-forge -c aihabitat
```

Then you can install the environment required for the project:

```bash
cd CanTest
pip install -r requirements.txt
```

## Data

We use [HM3D](https://aihabitat.org/datasets/hm3d/) as the scene dataset. You can download the splits we need by following the command below. Note that you need to submit an application to [Matterport](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data) before using it. For more information, please refer to [this link](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).

```bash
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_train_v0.2
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_val_v0.2
```

## Prompt
Our used prompt can be found in the directory: `case_gen/prompt`

# Reference
- https://github.com/HCPLab-SYSU/LH-VLN/
