import os
import json
from utils.demo_util import *
import gradio as gr

# --- Data Loading ---
episode_map, agent, model, task, split = load_episode()

global_state = {
    "current_episode": None,
    "current_step": 0,
    "env": load_environment(),
}

# --- Core Logic ---
def select_episode(instr_id):
    if not instr_id or instr_id == "file_not_found":
        return create_placeholder_image("Select an Episode"), create_placeholder_image(""), "Please select a valid episode.", "### 🚧 Status: Waiting"
    global_state["current_episode"] = episode_map[instr_id]
    global_state["current_step"] = 0
    return update_step_view()

def update_step_view():
    ep = global_state["current_episode"]
    step_idx = global_state["current_step"]
    scan = ep.get("scan", "unknown_scan")

    if step_idx == 0:
        info = [
            f"## Episode Start",
            f"**Instruction:** {ep.get('instruction', 'N/A')}",
            f"**Trajectory:** `[{ep['trajectory'][0][0]}]`",
            f"**History (Step 0):** {ep['history'][0]}"
        ]
        img_obs = load_image_observation(scan, ep['trajectory'][0])
        current_image = display_four_images_with_heading(img_obs, step_idx, ep.get("initial_heading", 0))
        return current_image, create_placeholder_image("Next action pending..."), "\n\n".join(info), "### 🚶 Navigation Status: Navigating..."

    detail_idx = step_idx - 1
    valid_trajectory = [ep['trajectory'][0]]
    current_heading = ep.get("initial_heading", 0)
    valid_actions_count = 0
    for i in range(detail_idx):
        prev_detail = ep["details"][i]
        if prev_detail.get("turned_angle") is not None:
            current_heading += prev_detail["turned_angle"]
            valid_actions_count += 1
            valid_trajectory.append(ep['trajectory'][valid_actions_count])

    current_viewpoint_id = valid_trajectory[-1]
    current_img_obs = load_image_observation(scan, current_viewpoint_id)
    current_image = display_four_images_with_heading(current_img_obs, step_idx-1, current_heading)

    detail = ep["details"][detail_idx]
    turned_angle = detail.get("turned_angle")
    is_valid_action = turned_angle is not None

    if is_valid_action:
        valid_actions_count += 1
        next_viewpoint_id = ep['trajectory'][valid_actions_count]
        valid_trajectory.append(next_viewpoint_id)
        next_heading = current_heading + turned_angle
        next_img_obs = load_image_observation(scan, next_viewpoint_id)
        next_image = display_four_images_with_heading(next_img_obs, step_idx, next_heading)
    else:
        next_image = display_four_images_with_heading(current_img_obs, -2, current_heading)

    info = [
        f"## Step {step_idx}",
        f"**Instruction:** {ep.get('instruction', 'N/A')}"
    ]

    if detail_idx < len(ep["llm_outputs"]):
        llm_output = ep["llm_outputs"][detail_idx]
        info.append(f"**LLM Output:**\n\n{llm_output}")

    action_info = f"- Turned Angle: `{turned_angle}`\n- Next Viewpoint ID: `{detail.get('viewpointID', 'N/A')}`"
    info.append(f"**Agent Action:**\n{action_info}")
    if not is_valid_action:
        info.append("**Status:** <span style='color:red;'>INVALID ACTION</span>")

    trajectory_str = " -> ".join([v[0] for v in valid_trajectory])
    info.append(f"**Full Trajectory:** `{trajectory_str}`")

    gt_path_str = " -> ".join(ep.get("gt_path", []))
    info.append(f"**GT Path:** `{gt_path_str}`")

    # Compare with GT path if available
    first_mismatch = None
    gt_path = ep.get("gt_path", [])
    for idx, (pred, gt) in enumerate(zip(valid_trajectory, gt_path)):
        if pred[0] != gt:
            first_mismatch = (idx, gt, pred[0])
            break

    if first_mismatch:
        mismatch_idx, expected_vid, actual_vid = first_mismatch
        current_viewpoint_id = valid_trajectory[mismatch_idx - 1][0] if mismatch_idx > 0 else valid_trajectory[0][0]
        neighbors = global_state['env']['navigable'][scan].get(current_viewpoint_id, {})

        # Ordered list of candidate viewpoint IDs
        ordered_candidates = list(neighbors.keys())  # already ordered by heading in data

        # Get index positions (1-based)
        try:
            expected_idx = ordered_candidates.index(expected_vid) + 1
        except ValueError:
            expected_idx = "❓"

        try:
            actual_idx = ordered_candidates.index(actual_vid) + 1
        except ValueError:
            actual_idx = "❓"

        info.append(
            f"<span style='color:red;'>⚠️ Agent deviated from GT path at step {mismatch_idx}: "
            f"expected `{expected_vid}` (option #{expected_idx}), got `{actual_vid}` (option #{actual_idx})</span>"
        )


    history_log = "\n".join([f"- {h.strip()}" for h in ep["history"][:step_idx + 1]])
    info.append(f"**History Log:**\n{history_log}")

    nav_status = "### 🚶 Navigation Status: Navigating..."
    if step_idx != 0:
        nav_status += (f"\n Current Viewpoint: {current_viewpoint_id} \n##### *Next Chosen Number: {detail.get('viewpointID', 'N/A')}*")
        
    if (detail_idx < len(ep["llm_outputs"]) and ep["llm_outputs"][detail_idx].strip().startswith("Action: Stop")) or \
       (step_idx == len(ep["details"])):
        info.append("\n---\n### Episode Finished")
        nav_status = "### 🏁 Navigation Status: <span style='color:green;'>Finished</span>"
        
        if is_valid_action:
            next_image = display_four_images_with_heading(next_img_obs, -1, next_heading)
        else:
            next_image = display_four_images_with_heading(current_img_obs, -1, current_heading)

    if "gt_path" in ep and (step_idx == len(ep["details"])):
        pred_traj = ep["trajectory"]
        gt_traj = ep["gt_path"]
        scan = ep["scan"]

        metrics = eval_single_episode(global_state["env"], pred_traj, gt_traj, scan)
        metric_str = format_metrics(metrics)
        info.append(metric_str)

    return current_image, next_image, "\n\n".join(info), nav_status

def next_step():
    ep = global_state["current_episode"]
    if ep is None:
        return create_placeholder_image("No episode selected"), create_placeholder_image(""), "Please select an episode first.", "### 🚧 Status: Waiting"
    if global_state["current_step"] < len(ep["details"]):
        global_state["current_step"] += 1
    return update_step_view()

def prev_step():
    if global_state["current_episode"] is None:
        return create_placeholder_image("No episode selected"), create_placeholder_image(""), "Please select an episode first.", "### 🚧 Status: Waiting"
    if global_state["current_step"] > 0:
        global_state["current_step"] -= 1
    return update_step_view()

# --- Gradio UI ---
episode_ids = list(episode_map.keys())
with gr.Blocks(theme=gr.themes.Default(), title="VLN Visualizer") as demo:
    gr.Markdown("# 🧭 VLN Result Visualizer")
    gr.Markdown(f"**Agent**: {agent}, **Model**: {model}, **Task**: {task}, **Split**: {split}")
    gr.Markdown("Select an episode to visualize the agent's step-by-step navigation process.")
    
    with gr.Row():
        dropdown = gr.Dropdown(choices=episode_ids, label="Select Episode ID", 
                               value=episode_ids[0] if episode_ids and episode_ids[0] != "file_not_found" else None, 
                               scale=2)
        with gr.Column():
            prev_btn = gr.Button("⬅️ Previous Step", scale=1, size='md')
            next_btn = gr.Button("Next Step ➡️", scale=1, size='md')

    status_text = gr.Markdown("### 🚧 Status: Waiting", elem_id="nav-status")
    image_output = gr.Image(type="pil", label="Current Viewpoint", interactive=False)
    next_image_output = gr.Image(type="pil", label="Next Viewpoint (if valid)", interactive=False)
    gr.Markdown("---")
    text_output = gr.Markdown(label="Episode Details")

    dropdown.change(fn=select_episode, inputs=dropdown, outputs=[image_output, next_image_output, text_output, status_text])
    next_btn.click(fn=next_step, outputs=[image_output, next_image_output, text_output, status_text])
    prev_btn.click(fn=prev_step, outputs=[image_output, next_image_output, text_output, status_text])
    demo.load(select_episode, inputs=dropdown, outputs=[image_output, next_image_output, text_output, status_text])

# Run the app
if __name__ == "__main__":
    demo.launch()
