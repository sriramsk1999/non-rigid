#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

# Example usage:
# ./multi_cloth_train.sh 0 cross_point_relative online
# ./multi_cloth_train.sh 1 scene_flow disabled dataset.multi_cloth.hole=single dataset.multi_cloth.size=100

GPU_INDEX=$1
MODEL_TYPE=$2
WANDB_MODE=$3
shift
shift
shift
COMMAND=$@

# scene flow model - no object-centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Training scene flow model with command: $COMMAND."

  WANDB_MODE=$WANDB_MODE python train_diff.py \
    model=df_base \
    dataset=pc_multi_cloth \
    dataset.scene=True \
    dataset.world_frame=True \
    wandb.group=pc_multi_cloth \
    resources.gpus=[${GPU_INDEX}] \
    $COMMAND
# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Training absolute flow model with command: $COMMAND."

  WANDB_MODE=$WANDB_MODE python train_diff.py \
    model=df_flow_cross \
    dataset=pc_multi_cloth \
    dataset.scene=False \
    dataset.world_frame=True \
    wandb.group=pc_multi_cloth \
    resources.gpus=[${GPU_INDEX}] \
    $COMMAND
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training relative flow model with command: $COMMAND."

  WANDB_MODE=$WANDB_MODE python train_diff.py \
    model=df_flow_cross \
    dataset=pc_multi_cloth \
    dataset.scene=False \
    dataset.world_frame=False \
    wandb.group=pc_multi_cloth \
    resources.gpus=[${GPU_INDEX}] \
    $COMMAND
# relative frame ghost point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training relative point model with command: $COMMAND."

  WANDB_MODE=$WANDB_MODE python train_diff.py \
    model=df_point_cross \
    dataset=pc_multi_cloth_point \
    dataset.world_frame=False \
    wandb.group=pc_multi_cloth \
    resources.gpus=[${GPU_INDEX}] \
    $COMMAND

else
  echo "Invalid model type."
fi