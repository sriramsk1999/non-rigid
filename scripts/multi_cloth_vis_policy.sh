#!/bin/bash

# This should take in 3 arguments:
# 1. model type
# 2. model checkpoint 
# 3. the rest of the arguments for the eval script

# Example usage:
# ./multi_cloth_vis_policy.sh cross_point_relative `CHECKPOINT`
# ./multi_cloth_vis_policy.sh scene_flow `CHECKPOINT` dataset.multi_cloth.hole=single dataset.multi_cloth.size=100

MODEL_TYPE=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

# scene flow model - no object-centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Evaluating scene flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  python vis_proc_cloth_policy.py \
    model=df_base \
    dataset=pc_multi_cloth \
    dataset.scene=True \
    dataset.world_frame=True \
    inference.action_full=True \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND

# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Evaluting absolute flow model at checkpoint $CHECKPOINT with command: $COMMAND."

python vis_proc_cloth_policy.py \
    model=df_flow_cross \
    dataset=pc_multi_cloth \
    dataset.scene=False \
    dataset.world_frame=True \
    inference.action_full=True \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND

# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Evaluating relative flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  python vis_proc_cloth_policy.py \
    model=df_flow_cross \
    dataset=pc_multi_cloth \
    dataset.scene=False \
    dataset.world_frame=False \
    inference.action_full=True \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND
# world frame cross point
elif [ $MODEL_TYPE == "cross_point_absolute" ]; then
  echo "Evaluating absolute point model at checkpoint $CHECKPOINT with command: $COMMAND."

  python vis_proc_cloth_policy.py \
    model=df_point_cross \
    dataset=pc_multi_cloth_point \
    dataset.scene=False \
    dataset.world_frame=True \
    inference.action_full=True \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND
# relative frame ghost point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Evaluating relative point model at checkpoint $CHECKPOINT with command: $COMMAND."

  python vis_proc_cloth_policy.py \
    model=df_point_cross \
    dataset=pc_multi_cloth_point \
    dataset.scene=False \
    dataset.world_frame=False \
    inference.action_full=True \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND

else
  echo "Invalid model type."
fi