# Installation #

This repository can be conveniently installed as a Python package, and used in downstream projects. Note: all of the up-to-date code is currently being kept under the ``articulated`` branch. Things will be merged to ``main`` in the future.

## Create conda environment and install dependencies ##

Before installing ``non-rigid``, create a conda environment with the required dependences in ``environment.yml``:

```
git clone https://github.com/ey-cai/non-rigid.git
git checkout articulated
cd non-rigid
conda env create --name ENVNAME --file=environment.yml
```
This is optional, but you may want to replace the ``name`` parameter in ``environment.yml`` with whatever you choose for ``ENVNAME`` for clarity.

## Install ``non-rigid`` ##

For now, the easiest thing to do is to install ``non-rigid`` in editable mode.

```
pip install -e .
```

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