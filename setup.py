#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ggkp",
    version="0.0.1",
    description="Experiments for GGKP.",
    author="Shafi Goldwasser, David F. Gruber, Adam Tauman Kalai, Orr Paradise",
    author_email="orrp@eecs.berkeley.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "wandb",
    ],
)