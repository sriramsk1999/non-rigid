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


# scene flow model - no object centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Training scene flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_base"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=True dataset.world_frame=True"
# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Training absolute flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=True"
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Training relative flow model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=flow dataset.scene=False dataset.world_frame=False"
# world frame cross point
elif [ $MODEL_TYPE == "cross_point_absolute" ]; then
  echo "Training absolute point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=True"
# relative frame cross point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Training relative point model with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=False"
# linear regression baseline
elif [ $MODEL_TYPE == "linear" ]; then
  echo "Training linear regression model with command: $COMMAND."

  MODEL_PARAMS="model=linear model.type=point"
  DATASET_PARAMS="dataset=proc_cloth dataset.type=point dataset.scene=False dataset.world_frame=False"
else
  echo "Invalid model type."
fi

WANDB_MODE=$WANDB_MODE python train_diff.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=proc_cloth \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND