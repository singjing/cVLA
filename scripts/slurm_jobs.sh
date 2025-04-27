#!/bin/bash
#SBATCH -p tflmb_gpu-rtx4090 # partition (queue)
#SBATCH --mem 32000 # memory pool for all cores (20GB)
#SBATCH -c 24 # number of cores
#SBATCH -a 1-6 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /ihome/argusm/lang/cVLA # Change working_dir
#SBATCH -o /work/dlclarge1/argusm-cvla/logs/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge1/argusm-cvla/logs/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

source ~/.bashrc

conda activate paligemma


# if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
#     python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --extra_run_name xyzrotvec-cam-1024xy 
#     exit $?
# fi

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder xyzrotvec-cam-128xy --extra_run_name xyzrotvec-cam-128xy 
    exit $?
fi

if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder  xyzrotvec-cam-256xy --extra_run_name xyzrotvec-cam-256xy 
    exit $?
fi

if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder xyzrotvec-cam-512xy128d --extra_run_name xyzrotvec-cam-512xy128d 
    exit $?
fi

if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder xyzrotvec-rbt-100 --extra_run_name xyzrotvec-rbt-100 
    exit $?
fi

if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder  xyzrotvec-rbt-128 --extra_run_name xyzrotvec-rbt-128 
    exit $?
fi

if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
    python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder  xyzrotvec-rbt-256 --extra_run_name xyzrotvec-rbt-256 
    exit $?
fi


# if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
#     python scripts/hf_image_condition.py --dataset_version mix30obj-8 --save_path /data/lmbraid19/argusm/models --eval_dataset double --extra_run_name xyzrotvec-cam-512xy 
#     exit $?
# fi

# if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
#     python scripts/hf_image_condition.py --dataset_version mix30obj --save_path /data/lmbraid19/argusm/models --eval_dataset double --action_encoder xyzrotvec-cam-512xy256d --extra_run_name xyzrotvec-cam-512xy256d 
#     exit $?
# fi

