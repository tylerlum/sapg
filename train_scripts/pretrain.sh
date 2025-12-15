#!/bin/bash

CUSTOM_EXPERIMENT_NAME="NEW_TARGET_VOLUME"
WANDB_GROUP="NEW_GAINS"

WANDB_ENTITY="kk837"
WANDB_PROJECT="FINAL_ASYMMETRIC_RUNS"
OBJECT_TYPE="cuboid"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}

python -m isaacgymenvs.train \
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
wandb_project=${WANDB_PROJECT} \
wandb_entity=${WANDB_ENTITY} \
wandb_activate=True \
wandb_group=${WANDB_GROUP} \
wandb_tags=[] \
++wandb_notes='' \
seed=0 \
experiment=00_${EXPERIMENT_NAME} \
hydra.run.dir=${HYDRA_RUN_DIR} \
task.env.object_type=${OBJECT_TYPE} \
task.env.useRelativeControl=False \
task.task.randomize=False \
task.env.stateList=["joint_pos","joint_vel","prev_action_targets","palm_pos","palm_rot","palm_vel","object_rot","object_vel","fingertip_pos_rel_palm","keypoints_rel_palm","keypoints_rel_goal","object_scales","closest_keypoint_max_dist","closest_fingertip_dist","lifted_object","progress","successes","reward"] \
task.env.obsList=["joint_pos","joint_vel","prev_action_targets","palm_pos","palm_rot","object_rot","fingertip_pos_rel_palm","keypoints_rel_palm","keypoints_rel_goal","object_scales"] \
task=AllegroKukaLSTMAsymmetric \
task.env.objectBaseSize=0.04 \
task.env.kukaActionsPenaltyScale=0.03 \
task.env.allegroActionsPenaltyScale=0.003 \
task.env.dofSpeedScale=2.5 \
task.env.controlFrequencyInv=1 \
task.env.armMovingAverage=0.1 \
task.env.handMovingAverage=0.1 \
task.env.episodeLength=600 \
task.env.successSteps=10 \
task.env.goalSamplingType=delta \
task.env.useObsDelay=False \
task.env.obsDelayMax=3 \
task.env.useActionDelay=False \
task.env.actionDelayMax=3 \
task.env.useObjectStateDelayNoise=False \
task.env.objectStateDelayMax=10 \
task.env.objectStateXyzNoiseStd=0.01 \
task.env.objectStateRotationNoiseDegrees=5.0 \
task.env.jointVelocityObsNoiseStd=0 \
# checkpoint=/share/portal/kk837/sapg/train_dir/FINAL_ASYMMETRIC_RUNS/NEW_GAINS/2.5_Speed_controlFreqInv_1_successSteps_10_delta_2025-12-09_19-47-41/runs/00_2.5_Speed_controlFreqInv_1_successSteps_10_delta_2025-12-09_19-47-41/last/model.pth \
# task.task.randomization_params.actor_params.object.scale.range=[0.999,1.001] \
# task.task.randomization_params.actor_params.allegro.scale.range=[0.999,1.001] \