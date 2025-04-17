#!/bin/bash
#SBATCH -p tflmb_gpu-rtx3090 # partition (queue)
#SBATCH --mem 32000 # memory pool for all cores (20GB)
#SBATCH -c 24 # number of cores
#SBATCH -a 1-12 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /home/bratulic/git_repos/robo/cVLA # Change working_dir
#SBATCH -o /work/dlclarge2/bratulic-cVLA/logs/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge2/bratulic-cVLA/logs/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

source ~/.bashrc

conda activate paligemma


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3  --no_augs --extra_run_name _randomSampling_newData_noAugs
    exit $?
fi

if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs --p_background 0.0
    exit $?
fi

if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_pBackground02
    exit $?
fi

if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3  --no_augs --extra_run_name _randomSampling_newData_noAugs_copy025 --p_copy 0.25
    exit $?
fi

if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_copy025 --p_background 0.0 --p_copy 0.25
    exit $?
fi

if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_pBackground02_copy025 --p_copy 0.25
    exit $?
fi

if [ 7 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3  --no_augs --extra_run_name _randomSampling_newData_noAugs_copy025_sort025_cameraSort --p_copy 0.25 --p_sort_by_l2_distance 0.25 
    exit $?
fi

if [ 8 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_copy025_sort025_cameraSort --p_background 0.0 --p_copy 0.25 --p_sort_by_l2_distance 0.25
    exit $?
fi

if [ 9 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_pBackground02_copy025_sort025_cameraSort --p_copy 0.25 --p_sort_by_l2_distance 0.25
    exit $?
fi


if [ 10 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3  --no_augs --extra_run_name _randomSampling_newData_noAugs_copy025_sort025_trajSort --p_copy 0.25 --p_sort_by_l2_distance 0.25 --sort_criteria trajectory_shape
    exit $?
fi

if [ 11 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_copy025_sort025_trajSort --p_background 0.0 --p_copy 0.25 --p_sort_by_l2_distance 0.25 --sort_criteria trajectory_shape
    exit $?
fi

if [ 12 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --conditioning trajectory --max_tokens 13 --dataset_version clevr_only --save_steps 250 --save_limit 10 --double_eval --batch_size_dev 3 --extra_run_name _randomSampling_newData_augs_pBackground02_copy025_sort025_trajSort --p_copy 0.25 --p_sort_by_l2_distance 0.25 --sort_criteria trajectory_shape
    exit $?
fi