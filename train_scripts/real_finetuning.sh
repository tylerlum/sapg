#!/bin/bash

export foldername="CUBOID_FINETUNING"
export filename_without_ext="RESTART_DR_NoObs_Slow"

# Use them to set wandb_group, experiment, and hydra_run_dir
export DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_GROUP=$foldername
# export EXPERIMENT=00_$filename_without_ext  # Must start with 00_ from code parsing
export EXPERIMENT=00_${filename_without_ext}_$DATETIME  # Add datetime to experiment name to avoid overwriting, end with it so datetime doesn't clog the real name
export HYDRA_RUN_DIR=./train_dir/allegro_kuka_reorientation/$WANDB_GROUP/$EXPERIMENT

# Select object type
export OBJECT_TYPE=cuboid

# Create paths to checkpoints
export CUBOID_PRETRAINED_CHECKPOINT=/share/portal/kk837/sapg/train_dir/allegro_kuka_reorientation/CUBOID_FINETUNING/00_DR_NoObs_Slow_2025-11-18_21-11-49/runs/00_DR_NoObs_Slow_2025-11-18_21-11-49/last/model.pth

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "filename_without_ext: $filename_without_ext"
echo "foldername: $foldername"
echo "WANDB_GROUP: $WANDB_GROUP"
echo "EXPERIMENT: $EXPERIMENT"
echo "HYDRA_RUN_DIR: $HYDRA_RUN_DIR"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

export CHECKPOINT=/share/portal/kk837/sapg/train_dir/allegro_kuka_reorientation/CUBOID_FINETUNING/00_DR_NoObs_Slow_2025-11-18_21-11-49/runs/00_DR_NoObs_Slow_2025-11-18_21-11-49/last/model.pth

# Run
command="\
python -m isaacgymenvs.train \
task=AllegroKukaLSTM \
task/env=reorientation \
++task.env.useSparseReward=False \
headless=True \
task.env.numEnvs=24576 \
train.params.config.minibatch_size=98304 \
multi_gpu=False \
train.params.config.good_reset_boundary=0 \
task.env.goodResetBoundary=0 \
train.params.config.use_others_experience=lf \
train.params.config.off_policy_ratio=1.0 \
train.params.config.expl_type=mixed_expl_learn_param \
train.params.config.expl_reward_type=entropy \
train.params.config.expl_coef_block_size=4096 \
train.params.config.expl_reward_coef_scale=0.005 \
train.params.network.space.continuous.fixed_sigma=coef_cond \
wandb_project=sapg_allegro_kuka_reorientation \
wandb_entity=kk837 \
wandb_activate=True \
wandb_group=$WANDB_GROUP \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=$EXPERIMENT \
hydra.run.dir=$HYDRA_RUN_DIR \
task.env.object_type=$OBJECT_TYPE \
task.env.use_fixed_set_of_goal_states=False \
task.env.use_fixed_init_object_pose=False \
task.env.capture_video=True \
task.env.armMovingAverage=0.1 \
task.env.handMovingAverage=0.1 \
task.env.dofSpeedScale=10.0 \
task.env.dofSpeedScaleFinal=2.5 \
task.env.useRelativeControl=False \
task.env.turn_off_object_vel_obs_slowly=True \
task.env.turn_off_extra_obs_slowly=True \
task.env.use_obs_dropout=True \
task.task.randomize=True \
task.env.allegroStiffness=5.0 \
task.env.allegroDamping=0.25 \
task.env.kukaStiffness=300.0 \
task.env.kukaDamping=20.0 \
checkpoint=${CHECKPOINT} \
task.env.curriculumSuccessRatio=0.1 \
task.env.forceScale=0.0 \
task.env.init_tyler_curriculum_scale=0.72 \
"

echo "Running command: $command"
$command
echo "Done running python script"

# Done
echo "DONE"