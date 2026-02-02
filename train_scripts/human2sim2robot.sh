#!/bin/bash
objectScaleNoiseMultiplierRange=[1.0,1.0]
forceConsecutiveNearGoalSteps=false
forceScale=2 
torqueScale=0
objectAngVelPenaltyScale=0.0

object_type="staples_open"
fixedGoalStatesJsonPath="/share/portal/kk837/sapg/dex_tool_bench/evaluation_trajectories/marker/staples_open/write_smiley_world_frame_min_z_0.6_downsampled_10.json"

CUSTOM_EXPERIMENT_NAME="${object_type}"
WANDB_GROUP="1x"

WANDB_ENTITY="kk837"
WANDB_PROJECT="Human2Sim2Robot"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}

CHECKPOINT=/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/HAMMER_DOWN_SWING/NoChanges_2026-01-18_03-43-03/runs/00_NoChanges_2026-01-18_03-43-03/last/model.pth

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
task=AllegroKukaLSTMAsymmetric \
task.env.objectScaleNoiseMultiplierRange=${objectScaleNoiseMultiplierRange} \
task.env.forceConsecutiveNearGoalSteps=${forceConsecutiveNearGoalSteps} \
task.env.forceScale=${forceScale} \
task.env.torqueScale=${torqueScale} \
task.env.objectAngVelPenaltyScale=${objectAngVelPenaltyScale} \
task.env.object_type=${object_type} \
task.env.use_fixed_set_of_goal_states=True \
task.env.fixedGoalStatesJsonPath=${fixedGoalStatesJsonPath} \
checkpoint=${CHECKPOINT} \