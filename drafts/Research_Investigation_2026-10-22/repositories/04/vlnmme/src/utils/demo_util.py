import os
import io
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils.data import load_hub_data, load_graph
from utils.eval import cal_cls, cal_dtw

ERROR_MARGIN = 3.0
OBS_FOLDER = "../data/marked_obs"
DIRECTIONS = ["left", "front", "right", "back"]

def load_episode(config_path="./configs/demo.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        agent = config['agent']
        model = config['model']
        task = config['task']
        split = config['split']
        RESULT_JSON_PATH = f"./outputs/{agent}_agent/{model}/{task}.{split}.json"

        try:
            with open(RESULT_JSON_PATH, 'r') as f:
                result_data = json.load(f)
            episode_map = {ep["instr_id"]: ep for ep in result_data}
        except FileNotFoundError:
            print(f"Error: Results file not found at {RESULT_JSON_PATH}")
            episode_map = {
                "file_not_found": {
                    "instruction": "Error: could not load result data.",
                    "history": [], "details": [], "llm_outputs": [], "trajectory": [], "scan": ""
                }
            }

        # Load original split file
        split_path = f"../data/{task}/{split}_enc.json"
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Original split file not found at: {split_path}")

        with open(split_path, 'r') as f:
            original_data = json.load(f)

        # Build path_id → path map
        _id_key = "id" if task == "REVERIE" else "path_id"
        pathid_to_path = {str(item[_id_key]): item["path"] for item in original_data}

        # Augment episode_map entries with matching path
        for instr_id, ep in episode_map.items():
            try:
                # Expected format: task_pathid_index, e.g., r2r_1622_0
                parts = instr_id.split("_")
                path_id = "_".join(parts[1:-1])  # second element is the numeric path_id
                if path_id in pathid_to_path:
                    ep["gt_path"] = pathid_to_path[path_id]
                else:
                    ep["gt_path"] = []
                    print(f"Warning: path_id {path_id} not found in original data.")
            except Exception as e:
                print(f"Failed to extract path_id from instr_id '{instr_id}': {e}")
                ep["gt_path"] = []

        return episode_map, agent, model, task, split
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading or parsing config file {config_path}: {e}")
        return None

def create_placeholder_image(text, width=1280, height=360):
    img = Image.new('RGB', (width, height), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_position = ((width - (text_bbox[2] - text_bbox[0])) / 2, (height - (text_bbox[3] - text_bbox[1])) / 2)
    d.text(text_position, text, fill=(255, 255, 255), font=font)
    return img

def display_four_images_with_heading(img_obs, step, heading_deg=0):
    normalized_heading = heading_deg % 360
    if normalized_heading < 0:
        normalized_heading += 360

    if (normalized_heading >= 315 or normalized_heading < 45):
        current_front_key = 'front'
    elif 45 <= normalized_heading < 135:
        current_front_key = 'right'
    elif 135 <= normalized_heading < 225:
        current_front_key = 'back'
    else:
        current_front_key = 'left'

    global_order_keys = ['left', 'front', 'right', 'back']
    idx_current_front = global_order_keys.index(current_front_key)
    display_keys = [
        global_order_keys[(idx_current_front - 1 + 4) % 4],
        global_order_keys[idx_current_front],
        global_order_keys[(idx_current_front + 1) % 4],
        global_order_keys[(idx_current_front + 2) % 4]
    ]

    # Fixed title order
    title_keys = ['left', 'front', 'right', 'back']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.9))
    if step > -1:
        fig.suptitle(f'Step: {step}', fontsize=20, y=0.97, fontweight='bold')
    elif step == -1:
        fig.suptitle(f'Finish!', fontsize=20, y=0.97, fontweight='bold', color="green")
    elif step == -2:
        fig.suptitle(f'Invalid Action!', fontsize=20, y=0.97, fontweight='bold', color="red")
    for i, (key, title_key) in enumerate(zip(display_keys, title_keys)):
        img_to_show = img_obs.get(key)
        axes[i].set_title(f"View: {title_key.capitalize()}", fontsize=18)  # Use fixed title order
        if img_to_show:
            axes[i].imshow(img_to_show)
        else:
            placeholder = create_placeholder_image(f"No '{title_key}' Image", width=img_obs['front'].width if img_obs.get('front') else 640)
            axes[i].imshow(placeholder)
        axes[i].axis('off')
    
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def load_image_observation(scan, viewpoint_id):
    obs = {}
    for direction in DIRECTIONS:
        filename = f"{scan}_{viewpoint_id[0]}_{direction}.png"
        path = os.path.join(OBS_FOLDER, filename)
        obs[direction] = Image.open(path).convert("RGB") if os.path.exists(path) else None
    return obs

def load_environment():
    navigable_data = load_hub_data("VLN-MME/MP3D", "navigable.json", save_dir="../data/MP3D")
    location_data = load_hub_data("VLN-MME/MP3D", "location.json", save_dir="../data/MP3D")
    scans = load_hub_data("VLN-MME/MP3D", "scans.txt", extension="txt", save_dir="../data/MP3D")

    graphs, shortest_paths, shortest_distance = load_graph(location_data, navigable_data, scans)

    environment = {
        "graphs": graphs,
        "shortest_path": shortest_paths,
        "shortest_distance": shortest_distance,
        "navigable": navigable_data,
        "location": location_data
    }

    return environment

def format_metrics(metrics):
    lines = [
        f"**Success (SR):** {'✅' if metrics['sr'] else '❌'}  (`{metrics['sr']:.2f}`)",
        f"**SPL:** `{metrics['spl'] * 100:.2f}%`",
        f"**Navigation Error:** `{metrics['nav_error']:.2f}` meters",
        f"**Oracle Error:** `{metrics['oracle_error']:.2f}` meters",
        f"**nDTW:** `{metrics['nDTW'] * 100:.2f}%`",
        f"**SDTW:** `{metrics['SDTW'] * 100:.2f}%`",
        f"**CLS:** `{metrics['CLS'] * 100:.2f}%`",
        f"**Trajectory Length:** `{metrics['trajectory_len']}` steps",
        f"**Trajectory Distance:** `{metrics['trajectory_dist']:.2f}` meters"
    ]
    return "### 🧮 Evaluation Metrics\n" + "\n\n".join(lines)

def eval_single_episode(env, pred_traj, gt_traj, scan):
    def _get_nearest(shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    shortest_distances = env['shortest_distance'][scan]

    flat_pred = sum(pred_traj, [])
    assert gt_traj[0] == flat_pred[0], 'Start position mismatch'

    nearest_position = _get_nearest(shortest_distances, gt_traj[-1], flat_pred)

    nav_error = shortest_distances[flat_pred[-1]][gt_traj[-1]]
    oracle_error = shortest_distances[nearest_position][gt_traj[-1]]

    trajectory_lengths = np.sum([
        shortest_distances[a][b] for a, b in zip(flat_pred[:-1], flat_pred[1:])
    ])
    gt_lengths = np.sum([
        shortest_distances[a][b] for a, b in zip(gt_traj[:-1], gt_traj[1:])
    ])

    success = float(nav_error < ERROR_MARGIN)
    spl = success * gt_lengths / max(trajectory_lengths, gt_lengths, 0.01)
    oracle_success = float(oracle_error < ERROR_MARGIN)

    dtw_scores = cal_dtw(shortest_distances, flat_pred, gt_traj, success, ERROR_MARGIN)
    cls_score = cal_cls(shortest_distances, flat_pred, gt_traj, ERROR_MARGIN)

    return {
        "nav_error": nav_error,
        "oracle_error": oracle_error,
        "trajectory_len": len(flat_pred) - 1,
        "trajectory_dist": trajectory_lengths,
        "sr": success,
        "oracle_sr": oracle_success,
        "spl": spl,
        "nDTW": dtw_scores["nDTW"],
        "SDTW": dtw_scores["SDTW"],
        "CLS": cls_score
    }
