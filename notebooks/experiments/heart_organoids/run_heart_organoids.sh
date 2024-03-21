#!/bin/bash

source ~/.bashrc
conda activate /home/ueltzhoe/conda-py310/
cd /g/stegle/ueltzhoe/bicycle/notebooks/experiments/heart_organoids
python bootstrapped_cluster_run_heart_organoids.py $1
