# CVLA

## General developement setup


## Generate the dataset
Create a conda env ( try default, otherwise downgrade to 3.10)
```
conda create env -n maniskill
conda activate maniskill

cd project_dir/ManiSkill/mani_skill/examples
python gen_dataset.py -e "ClevrMove-v1"   --render-mode="rgb_array" 
python gen_dataset_cleanup.py
# this generates /tmp/clevr-act-1.zip
```

### Other simultation envs
```
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human"
```

## Training


### First start by copying dataset to server
```
# on local machine
cp /tmp/clevr-act-1.zip /data/lmbraid19/argusm/datasets

# on cluster machin 
mkdir /tmp/clevr-act-1
cd /tmp/clevr-act-1
cp /misc/lmbraid19/argusm/datasets/clevr-act-1.zip .
unzip -q clevr-act-1.zip
cp dataset/_annotations.test.jsonl dataset/_annotations.valid.jsonl
```

### Running training

Create a conda env (python version 3.12 because of tensorboard)
```
conda create env -n paligemma python=3.12
```

1. Use VS Code, set up remote connection to e.g. bud
2. Install Python and Jupyter packages on remote and set notebook to `paligemma`
3. Run notebook and follow initial instructions for Kaggle / model access
3. Get a kaggle key and create `~/.kaggle/kaggle.json` with `{"username":"XXX","key":"YYY"}`
3. Run `./cVLA/finetune_paligemma_robotflow.ipynb`