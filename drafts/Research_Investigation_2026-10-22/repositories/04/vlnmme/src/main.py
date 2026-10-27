import os
import json
import numpy as np
import torch
import random
import time
import argparse

from agents import get_agents
from omegaconf import OmegaConf
from runner.loader import create_environment, create_val_env
from utils.common_utils import setup_logger, write_to_record_file

def setup_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment runner with OmegaConf")

    parser.add_argument(
        "--config_dir", 
        type=str, 
        default="configs/experiment.yaml",
        help="Path to the experiment config file"
    )

    args = parser.parse_args()

    # Load the default configuration
    default_config_path = 'configs/default.yaml'
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")
    default_config = OmegaConf.load(default_config_path)

    # Load the experiment-specific configuration
    if not os.path.exists(args.config_dir):
        raise FileNotFoundError(f"Experiment config file not found: {args.config_dir}")
    experiment_config = OmegaConf.load(args.config_dir)

    # Merge default config with experiment config
    config = OmegaConf.merge(default_config, experiment_config)

    return config

def valid(config, val_envs, logger):
    for env_name, env in val_envs.items():
        logger.info(f"Evaluating split: {env_name}")

        agent_class = get_agents(config.agent.type, logger)
        agent = agent_class(env_name, env, config, logger)

        # Evaluate the split and track time
        start_time = time.time()
        agent.test()

        logger.info(f"{env_name}, cost time: %.2fs" % (time.time()-start_time))

        preds = agent.get_results(detailed_output=False)

        score_summary, _ = env.eval_metric(preds, logger)
        loss_str = "Env name: %s" % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)

        record_file = os.path.join(config.experiment.output_dir, config.agent.type, config.experiment.model, 'valid.txt')
        write_to_record_file(loss_str+'\n', record_file)

def valid_from_file(config, val_envs, logger):
    for env_name, env in val_envs.items():
        logger.info(f"Evaluating split: {env_name} from file")

        # Evaluate the split
        file_path = os.path.join(
            config.experiment.output_dir, 
            config.agent.type, 
            config.experiment.model, 
            f'{env_name}.json'
        )

        with open(file_path, 'r') as f:
            data = json.load(f)
            preds = [
                {'instr_id': item['instr_id'], 'trajectory': item['trajectory']}
                for item in data
            ]

        score_summary, metric = env.eval_metric(preds, logger)
        
        print(f"{'instr_id':<20} {'Success':<10} {'Oracle_Success':<15}")
        print("-" * 45)

        for i in range(len(metric['instr_id'])):
            instr_id = metric['instr_id'][i]
            success = metric['success'][i]
            oracle_success = metric['oracle_success'][i]

            print(f"{instr_id:<20} {success:<10} {oracle_success:<15}")
        
        
        loss_str = "Env name: %s" % env_name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)

        record_file = os.path.join(config.experiment.output_dir, config.agent.type, config.experiment.model, 'valid.txt')
        write_to_record_file(loss_str+'\n', record_file)


def main():
    ############################################################
    # Parse arguments and load configuration
    ############################################################
    config = parse_args()

    # Setup logging
    output_path = os.path.join(config.experiment.output_dir, config.agent.type)
    os.makedirs(output_path, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(output_path, f'{config.agent.type}.log'), rank=config.distributed.rank)

    # Log configuration (from OmegaConf)
    logger.info(f'********************** Start logging **********************')
    
    # Random seed setting
    setup_seeds(seed=config.experiment.seed)


    ############################################################
    # Set up data loaders and agents
    ############################################################
    logger.info('********************** Setting up dataloaders **********************')

    environment = create_environment(config, logger)
    logger.info("Finished building MP3D environment")

    val_envs = create_val_env(config, logger, environment)
    logger.info("Finished building data batch env")

    ############################################################
    # Validation 
    ############################################################
    if config.experiment.valid_from_file:
        valid_from_file(config, val_envs, logger)
    else:
        valid(config, val_envs, logger)


if __name__ == '__main__':
    main()
    