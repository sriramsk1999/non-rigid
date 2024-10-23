# Installation #

This repository can be conveniently installed as a Python package, and used in downstream projects. Note: all of the up-to-date code is currently being kept under the ``articulated`` branch. Things will be merged to ``main`` in the future.

## Create conda environment and install dependencies ##

For now, all of the up-to-date TAX3D code is in the ``articulated`` branch.
```
git clone https://github.com/ey-cai/non-rigid.git
cd non-rigid
git checkout articulated
conda create --name ENVNAME python=3.9 pip==23.3.1 setuptools==65.5
```
Before installing ``non-rigid``, you'll need to install versions of PyTorch, PyTorch Geometric, and PyTorch3D. We've provided specific GPU versions in ``requirements-gpu.txt``:
```
pip install -r requirements-gpu.txt
```

## Install DEDO and 3D Diffusion Policy ##

```
cd third_party/dedo
pip install -e .
cd ../..

cd third_party/3D-Diffusion-Policy/3D-Diffusion-Policy
pip install -e .
cd ../../..
```
Note: the TAX3D repo contains significant changes to the underlying code in both the DEDO and 3D Diffusion Policy repository (refer to the READMEs in the respective ``third_party/`` directories, will update in the future).


## Install ``non-rigid`` ##

For now, the easiest thing to do is to install ``non-rigid`` in editable mode. This should install all of the additional required dependencies, as listed in ``pyproject.toml``.

```
pip install -e .
```
And you're done!

# Training Models #
To train a model, run:
```
./multi_cloth_train.sh [GPU_INDEX] [MODEL_TYPE] [WANDB_MODE]
```
For example, to train a TAX3D-CD model and log results to WandB, run:
```
./multi_cloth_train.sh 0 cross_flow_relative online
```
Note: you may have to update the ``data_dir`` parameter in ``configs/dataset/proc_cloth.yaml`` to properly reflect the directory where your data is stored. This can also be done from the command line:
```
./multi_cloth_train.sh 0 cross_flow_relative online dataset.data_dir=[PATH_TO_YOUR_DATASET]
```

# Running Evaluations #
To get coverage metrics, run:
```
./multi_cloth_eval.sh [GPU_INDEX] [MODEL_TYPE] [CHECKPOINT] coverage=True
```
For example:
```
./multi_cloth_eval.sh 0 cross_flow_relative sfr4r4hs coverage=True
```
Note: you may have to update the ``data_dir`` parameter in ``configs/dataset/proc_cloth.yaml`` to properly reflect the directory where your data is stored. This can also be done from the command line:
```
./multi_cloth_eval.sh 0 cross_flow_relative sfr4r4hs coverage=True dataset.data_dir=[PATH_TO_YOUR_DATASET]
```

# Running Policy Evaluations #
TODO