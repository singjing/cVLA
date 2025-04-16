#!/bin/bash
#SBATCH -p tflmb_gpu-rtx4090 # partition (queue)
#SBATCH --mem 32000 # memory pool for all cores (20GB)
#SBATCH -c 24 # number of cores
#SBATCH -a 1-2 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /home/bratulic/git_repos/robo/cVLA # Change working_dir
#SBATCH -o /work/dlclarge2/bratulic-cVLA/logs/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge2/bratulic-cVLA/logs/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

source ~/.bashrc

conda activate paligemma


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --p_background 0.35 --save_steps 250 --save_limit 10 --eval_dataset sim --batch_size_dev 3 --max_steps 2000 
    exit $?
fi

[ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --p_background 0.35 --save_steps 250 --save_limit 10 --eval_dataset sim --batch_size_dev 3 --max_steps 6000 
    exit $?