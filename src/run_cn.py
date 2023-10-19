import argparse
import random

import wandb
from tqdm import tqdm

from src.learners.random_permutation import RandomPermutationLearner

from src.envs.commonnonsense import CommonNonsenseModel

import numpy as np

import logging


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for the experiment',
        required=True
    )
    parser.add_argument(
        '--train_size',
        type=int,
        help='Num samples for training',
        required=True
    )
    parser.add_argument(
        '--valid_size',
        type=int,
        help='Num samples for validation',
        required=True
    )
    parser.add_argument(
        '--P_size',
        type=int,
        help='Size of the set P',
        required=True
    )
    parser.add_argument(
        '--T_size',
        type=int,
        help='Size of the set T',
        required=True
    )
    parser.add_argument(
        '--alpha',
        type=float,
        help='Probability of dropping a sentences (float between 0.0 and 1.0)',
        required=True
    )
    parser.add_argument(
        '--num_translators',
        type=int,
        help='Size of the translator family',
        required=True
    )
    parser.add_argument(
        '--num_translators_to_valid',
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
    positive_params = ['train_size', 'valid_size', 'P_size', 'T_size', 'num_translators']
    for param in positive_params:
        if config[param] <= 0:
            raise ValueError(f'{param} must be a positive integer')
    if config['alpha'] < 0.0 or config['alpha'] >= 1.0:
        raise ValueError(f'alpha must be in [0.0, 1.0)')
    if config['train_size'] + config['valid_size'] > config['T_size']:
        raise ValueError(f'Sum of train and validation sizes should be smaller than T.')


def main():
    args = parse()
    config = vars(args)
    validate_config(config)

    random.seed(config['seed'])
    np.random.seed(config['seed'])

    if not config['no_wandb']:
        group_name = f"cn_P{config['P_size']},T{config['T_size']},alpha{config['alpha']},ns{config['train_size']},nt{config['num_translators']}"
        run_name = f"{group_name},seed{config['seed']}"
        run = wandb.init(
            project='ggkp',
            config=config,
            group=group_name,
            name=run_name
        )
    env = CommonNonsenseModel(
        P_size=config['P_size'],
        T_size=config['T_size'],
        alpha=config['alpha']
    )

    learner = RandomPermutationLearner(env, config['num_translators'])
    ground_truth = random.choice(learner.translators)
    env.set_ground_truth(ground_truth)

    # Generate train and validation samples.
    train_size, valid_size = config['train_size'], config['valid_size']
    if train_size + valid_size > len(env.Mu):
        remaining = len(env.Mu) - train_size
        logging.warning(f'Sum of train and valid sizes was greater than Mu. Reducing valid size to {remaining}')
        config['valid_size'] = remaining
        valid_size = remaining
    samples = random.sample(env.Mu, train_size + valid_size)
    train_samples, valid_samples = samples[:train_size], samples[train_size:]

    for x in tqdm(train_samples, desc='Training samples'):
        num_plausible = len(learner.plausible_translators)
        metrics = {'num_plausible': num_plausible}
        if num_plausible < config['num_translators_to_valid'] and learner.validation_errors is None:
            logging.info('Computing validation errors')
            learner.initialize_validation_errors(valid_samples)
        learner.train_iter(x)
        if not config['no_wandb']:
            metrics['num_plausible'] = num_plausible
            if learner.validation_errors is not None:
                metrics.update({
                    'avg_error': np.mean(learner.validation_errors),
                    'std_error': np.std(learner.validation_errors),
                    'max_error': max(learner.validation_errors)
                })
            else:
                metrics.update({
                    'avg_error': 1,
                    'std_error': 0,
                    'max_error': 1
                })
            wandb.log(metrics)
    if not config['no_wandb']:
        run.log_code()


if __name__ == "__main__":
    main()
