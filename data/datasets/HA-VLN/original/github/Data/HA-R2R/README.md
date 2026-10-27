### HA-R2R Dataset

🚀🚀🚀 [**Download Here**](https://drive.google.com/drive/folders/1ci6Fwxa86uBbcVQy8qJlJRhgLjj12d1u?usp=drive_link)

Instruction Examples Table presents four instruction examples from the **Human-Aware Room-to-Room (HA-R2R) dataset**. These cases include various scenarios such as:
- **Multi-human interactions** (e.g., 1, 2, 3),
- **Agent-human interactions** (e.g., 1, 2, 3),
- **Agent encounters four or more humans** (e.g., 3),
- **No humans encountered** (e.g., 4).

These examples illustrate the diversity of **human-aligned navigation instructions** that challenge the agent in our task.

### Instruction Examples Table

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
