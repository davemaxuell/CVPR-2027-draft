🚀🚀🚀 [**Download Here**](https://drive.google.com/drive/folders/1WrdsRSPp-xJkImZ3CnI7Ho90lnhzp5GR?usp=sharing)

# HAPS Dataset 2.0

In real-world scenarios, human motion typically adapts and interacts with the surrounding region. The proposed **Human Activity and Pose Simulation (HAPS) Dataset 2.0** improves upon [**HAPS 1.0**](https://github.com/lpercc/HA3D_simulator/) by making the following enhancements:  
1. *Refining and diversifying human motions.*  
2. *Providing descriptions closely tied to region awareness.*   

HAPS 2.0 mitigates the limitations of existing human motion datasets by identifying **26 distinct regions** across **90 architectural scenes** and generating **486 human activity descriptions**, encompassing both **indoor and outdoor environments**. These descriptions, validated through **human surveys** and **quality control using ChatGPT-4**, include realistic actions and region annotations (e.g., *"workout gym exercise: An individual running on a treadmill"*).

The [**Motion Diffusion Model (MDM)**](https://guytevet.github.io/mdm-page/) converts these descriptions into **486 detailed 3D human motion models** $\mathbf{H}$[^1] using the **SMPL model**, each transformed into a **120-frame motion sequence** $\mathcal{H}$.  

Each **120-frame SMPL mesh sequence** $\mathcal{H} = \langle h_1, h_2, \ldots, h_{120} \rangle$ details **3D human motion and shape information** through the **SMPL model**.

---

[^1]: $\mathbf{H}\ = \mathbb{R}^{486 \times 120 \times(10+72+6890 \times 3)}$, representing **486 models**, each with **120 frames**, including **shape, pose, and mesh vertex parameters**.

# HA-R2R Dataset

Instruction Examples Table presents four instruction examples from the **Human-Aware Room-to-Room (HA-R2R) dataset**. These cases include various scenarios such as:
- **Multi-human interactions** (e.g., 1, 2, 3),
- **Agent-human interactions** (e.g., 1, 2, 3),
- **Agent encounters four or more humans** (e.g., 3),
- **No humans encountered** (e.g., 4).

These examples illustrate the diversity of **human-aligned navigation instructions** that challenge the agent in our task.

## Instruction Examples Table

| **Instruction Example** |
|-------------------------|
| **1.** Exit the library and turn left. As you proceed straight ahead, you will enter the bedroom, **where you can observe a person actively searching for a lost item, perhaps checking under the bed or inside drawers**. Continue moving forward, **ensuring you do not disturb his search**. As you pass by, **you might see a family engaged in a casual conversation on the porch or terrace**, **be careful not to bump into them**. Maintain your course until you reach the closet. Stop just outside the closet and await further instructions. |
| **2.** Begin your path on the left side of the dining room, **where a group of friends is gathered around a table, enjoying dinner and exchanging stories with laughter**. As you move across this area, **be cautious not to disturb their gathering**. The dining room features a large table and chairs. Proceed through the doorway that leads out of the dining room. Upon entering the hallway, continue straight and then make a left turn. As you walk down this corridor, you might notice framed pictures along the walls. The sound of laughter and conversation from the dining room may still be audible as you move further away. Continue down the hallway until you reach the entrance of the office. Here, **you will observe a person engaged in taking photographs, likely focusing on capturing the view from a window or an interesting aspect of the room**. Stop at this point, ensuring you are positioned at the entrance without obstructing the photographer's activity. |
| **3.** Starting in the living room, **you can observe an individual practicing dance moves, possibly trying out new steps**. As you proceed straight ahead, **you will pass by couches where a couple is engaged in a quiet, intimate conversation, speaking softly to maintain their privacy**. Continue moving forward, ensuring you navigate around any furniture or obstacles in your path. As you transition into the hallway, **notice another couple enjoying a date night at the bar, perhaps sharing drinks and laughter**. **Maintain a steady course without disturbing them**, keeping to the right side of the hallway. Upon reaching the end of your path, you will find yourself back in the living room. Here, **a person is checking their appearance in a hallway mirror, possibly adjusting their attire or hair**. Stop by the right candle mounted on the wall, ensuring you are positioned without blocking any pathways. |
| **4.** Begin by leaving the room and turning to your right. Proceed down the hallway, be careful of any human activity or objects along the way. As you continue, look for the first doorway on your right. Enter through this doorway and advance towards the shelves. Once you reach the vicinity of the shelves, come to a halt and wait there. During this movement, avoid any obstacles or disruptions in the environment. |

 **purple-highlighted instructions** relate to **human movements**, and **blue-highlighted instructions** are associated with **agent-human interactions**.

---

## HA-R2R Instruction Generation

To generate new instructions for the **HA-R2R dataset**, we employ **ChatGPT-4o** and **LLaMA-3-8B-Instruct** to **contextually enrich and expand scene information** based on the original instructions from the **R2R-CE dataset**.

### **Few-Shot Prompting Approach**
Our approach utilizes a **few-shot template prompt**, consisting of:
- **A system prompt** 
- **A set of few-shot examples** 

The **system prompt** primes the LLMs with the **context and requirements** for generating **navigation instructions** in human-populated environments. It outlines the **desired characteristics**, such as:
- **Relevance** to the navigation task,
- **Integration of human activities and agent interactions**, and
- **Precision in describing environmental details**.

The **few-shot examples** serve as **guidelines** for how the instructions should be structured, demonstrating:
- **Incorporation of human activities**,
- **Use of relative position information**, and
- **Integration with original navigation instructions**.

For instance, one **example** includes:
> *“You will notice someone quietly making a phone call, so please remain quiet as you move.”*

---

### **Iterative Refinement Process**
Initially, the models produced **irrelevant or subjective content** and lacked sufficient **detail about human activities**. To improve this:
1. We **reviewed outputs** and identified discrepancies.
2. We **refined the system prompt** to emphasize **neutral tone** and **avoid subjective descriptions**.
3. We **updated few-shot examples** to demonstrate **desired level of detail**.
4. Through multiple iterations, we ensured that **scene-enriched instructions** remained coherent and relevant to **HA-R2R**.

---

### **LLM Prompt for HA-R2R Instruction Generation**
#### **System Prompt**
```json
{
  "system": "You are tasked with generating an objective and detailed description of a path and the related scene information based on the given input. Focus solely on observable entities and actions in the scene, avoiding descriptions of smell, atmosphere, or subjective feelings. Ensure clarity and precision in object and human activity descriptions.",
  "examples": "... //examples in Listing 4",
  "user": "Input:\n[instruction_text]: \"...\";\n[start]: \"...\";\n[during]: \"...\";\n[end]: \"...\";\n\nPlease provide an objective, step-by-step description of the path, focusing on the objects and human activities observed at each stage of the navigation (beginning, during, and end). Expand your description by including character actions, objective descriptions of objects, and details of the movement process. Your description should maintain a neutral tone and avoid embellishments. Please simulate the robot's movement along the path, while incorporating possible connections or interactions between the robot, objects, and human activities.\n\nOutput: ... //generation"
}

{
  "example_1": {
    "input": {
      "instruction_text": "Go through the doorway to the hall. Go down the hall past the landing. Take the first door on the right and stop in the doorway.",
      "start": "other room: A person using the room for a quiet phone call.",
      "during": "other room: A person using the room for a quiet phone call.",
      "end": "hallway: An individual cleaning or vacuuming."
    },
    "output": "You should start from the room. Before you walk through the doorway into the hallway, you will notice someone quietly making a phone call, so please remain quiet as you move. Then, proceed down the hallway past the landing. As you continue through the corridor, you will see a responsible cleaner using a vacuum to tidy up. Finally, your destination is at the end of the hallway, enter the first door on the right and stop in the doorway."
  }
}
```

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
