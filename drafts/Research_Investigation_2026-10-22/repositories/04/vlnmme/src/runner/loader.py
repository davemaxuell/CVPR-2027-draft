import copy
from logging import Logger
from omegaconf import DictConfig

from utils.data import load_hub_data, load_graph
from runner.feature_db import create_observation_db
from tasks import get_data


def create_environment(config, logger):
    logger.info("Loading simulation envrionment - MP3D ...")

    navigable_data = load_hub_data("VLN-MME/MP3D", "navigable.json", save_dir="../data/MP3D")
    location_data = load_hub_data("VLN-MME/MP3D", "location.json", save_dir="../data/MP3D")
    scans = load_hub_data("VLN-MME/MP3D", "scans.txt", extension="txt", save_dir="../data/MP3D")

    load_hub_data("VLN-MME/MP3D", "caption.json", save_dir="../data/MP3D")

    graphs, shortest_paths, shortest_distance = load_graph(location_data, navigable_data, scans)

    logger.info(f"Loaded {len(scans)} scans")

    environment = {
        "graphs": graphs,
        "shortest_path": shortest_paths,
        "shortest_distance": shortest_distance,
        "navigable": navigable_data,
        "location": location_data,
    }
    
    if "OBJECTNAV" in config.task.val_source:
        object_location_data = load_hub_data("VLN-MME/MP3D", "object_location.json", save_dir="../data/MP3D")
        environment["object_location"] = object_location_data

    return environment


def create_val_env(
    config: DictConfig, 
    logger: Logger, 
    environment,
):
    task_cfg = copy.deepcopy(config.task)

    # Create image observation
    image_obs_db = create_observation_db(config, logger)

    val_envs = {}

    # Load datasets
    for task in task_cfg.val_source:
        splits = task_cfg.eval_splits[task]
        if isinstance(splits, str):
            splits = [splits]

        for split in splits:
            # Build dataset class
            dataset = get_data(
                config=config, 
                split=split, 
                environment=environment, 
                obs_db = image_obs_db,
                logger=logger, 
                task=task
            )

            env_name = f"{task}.{split}"
            val_envs[env_name] = dataset

    return val_envs