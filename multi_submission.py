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
def submit_batch_job(arguments, dataset):

    # Set dynamic parameters for the batch job as environment variables
    # But dont forget to add the os.environ to the new environment variables otherwise the PATH is not found
    env = {
        **os.environ,
        "EXP_PARAMS": f"-S dataset.subset={dataset['subset']} -S dataset.split={dataset['split']}",
    }

    # For debugging and local runs
    
    if shutil.which('sbatch') is None:
        print(f"SLURM not available. Would submit job with:")
        print(f"  Dataset Subset: {dataset['subset']}")
        print(f"  Dataset Split: {dataset['split']}")
        print(f"  Arguments: {' '.join(arguments)}")
        print(f"  Environment: EXP_PARAMS={env['EXP_PARAMS']}")
        subprocess.run(['bash', '-c', f'./exp_workflow.sh {" ".join(arguments)}'], env=env)
        return
    
    # Run sbatch command with the environment variables as bash! subprocess! command (otherwise module not found)
    subprocess.run(['/usr/bin/bash', '-c', f'sbatch slurm_job.sh {" ".join(arguments)}'], env=env)

if __name__ == "__main__":

    arguments = sys.argv[1:]

    datasets = [{'subset': 'HSN_xc', 'split': 'train'}]

    for dataset in datasets:
        submit_batch_job(arguments, dataset)

    # test_split_list = [0.2, 0.3]
    # batch_size_list = [2048, 4096]
    # # Iterate over a cartesian product parameter grid of the test_split and batch_size lists
    # for test_split, batch_size in itertools.product(test_split_list, batch_size_list):
    #     submit_batch_job(arguments, test_split, batch_size)
