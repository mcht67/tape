#!./venv/bin/python

# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

import itertools
import subprocess
import os
import sys
import shutil

# Submit experiment for hyperparameter combination
def submit_batch_job(arguments, dataset, feature, epochs):

    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S dataset.subset={dataset['subset']} -S dataset.split={dataset['split']} -S train.features={feature} -S train.epochs={epochs}",
        "DEFAULT_DIR": os.getcwd(),
        "TUSTU_SYNC_INTERVAL": None
    }

    # For debugging and local runs
    
    if shutil.which('sbatch') is None:
        print(f"SLURM not available. Would submit job with:")
        print(f"  Dataset Subset: {dataset['subset']}")
        print(f"  Dataset Split: {dataset['split']}")
        print(f'  Feature: {feature}')
        print(f"  Epochs: {epochs}")
        print(f"  Arguments: {' '.join(arguments)}")
        print(f"  Environment: EXP_PARAMS={env['EXP_PARAMS']}")

        # Run DVC experiment directly
        cmd = 'dvc exp run $EXP_PARAMS'
        print(cmd)
        subprocess.run(cmd, shell=True, env=env)
        return
    
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    dataset_list = [{'subset': 'HSN_xc', 'split': 'train'}]
    features_list = ['perch_8_embeddings'] #, 'yamnet_embeddings']
    epochs_list = [1]

    for features, dataset, epochs in itertools.product(features_list, dataset_list, epochs_list) :
        submit_batch_job(arguments, dataset, features, epochs)

    # test_split_list = [0.2, 0.3]
    # batch_size_list = [2048, 4096]
    # # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    # for test_split, batch_size in itertools.product(test_split_list, batch_size_list):
    #     submit_batch_job(arguments, test_split, batch_size)
