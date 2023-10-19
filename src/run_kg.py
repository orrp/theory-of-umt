import argparse
import logging
import random

import numpy as np
import wandb
from tqdm import tqdm

from src.envs.knowledge_graph import KnowledgeGraphModel
from src.learners.knowledge_graph import KnowledgeGraphLearner


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for the experiment',
        required=True
    )
    parser.add_argument(
        '--n_target',
        type=int,
        help='Number of nodes in the target graph',
        required=True
    )
    parser.add_argument(
        '--n_source',
        type=int,
        help='Number of nodes in the target graph',
        required=True
    )
    parser.add_argument(
        '--alpha',
        type=float,
        help='Agreement coefficient (float between 0.0 and 1.0)',
        required=True
    )
    parser.add_argument(
        '--p',
        type=float,
        help='Probability of having an edge (float between 0.0 and 1.0)',
        required=True
    )
    parser.add_argument(
        '--n_top_translators',
        type=int,
        help='Maximum number of translators to validate',
        required=True
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Do not log to WandB',
    )
    args = parser.parse_args()
    return args


def validate_config(config):
    positive_params = ['n_target', 'n_source', 'n_top_translators']
    for param in positive_params:
        if config[param] <= 0:
            raise ValueError(f'{param} must be a positive integer')
    floats_in_0_1 = ['alpha', 'p']
    for param in floats_in_0_1:
        if config[param] < 0.0 or config[param] > 1.0:
            raise ValueError(f'{param} must be in [0.0, 1.0]')


def main():
    args = parse()
    config = vars(args)
    validate_config(config)

    random.seed(config['seed'])
    np.random.seed(config['seed'])

    if not config['no_wandb']:
        group_name = f"kg_ntgt{config['n_target']},nsrc{config['n_source']},alpha{config['alpha']},p{config['p']}"
        run_name = f"{group_name},seed{config['seed']}"
        run = wandb.init(
            project='ggkp',
            config=config,
            group=group_name,
            name=run_name
        )
    env = KnowledgeGraphModel(
        n_target=config['n_target'],
        n_source=config['n_source'],
        p=config['p'],
        alpha=config['alpha'],
    )
    metrics_logger = wandb.log if not config['no_wandb'] else logging.info
    learner = KnowledgeGraphLearner(env, config['n_top_translators'])
    ground_truth = random.choice(learner.translators)
    env.set_ground_truth(ground_truth)

    metrics = {}
    for x1, x2 in tqdm(env.edges_in_random_order, desc='Training samples'):
        learner.train_iter(x1, x2)
        for k, errors in learner.validation_errors.items():
            metrics[f'{k}_avg_error'] = np.mean(errors)
            if config['n_top_translators'] > 1:  # If it's just one then the following are trivial.
                metrics[f'{k}_std_error'] = np.std(errors)
                metrics[f'{k}_max_error'] = max(errors)
                metrics[f'{k}_min_error'] = min(errors)
        metrics_logger(metrics)
    # Finally, pad the run with the final metrics so that all seeds have the same length...
    possible_num_edges = config['n_source'] ** 2
    for _ in range(possible_num_edges - len(env.edges_in_random_order)):
        metrics_logger(metrics)


if __name__ == "__main__":
    main()
