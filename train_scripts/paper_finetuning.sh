#!/bin/bash
objectScaleNoiseMultiplierRange=[0.9,1.1]
forceConsecutiveNearGoalSteps=true
forceScale=20
torqueScale=2
objectAngVelPenaltyScale=0

CUSTOM_EXPERIMENT_NAME="FT_All_Changes"
WANDB_GROUP="FINETUNE_3x"

WANDB_ENTITY="kk837"
WANDB_PROJECT="PAPER_RUNS"

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME}_$DATETIME"
HYDRA_RUN_DIR=./train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}

CHECKPOINT=/share/portal/kk837/sapg/train_dir/PAPER_RUNS/FINETUNE_2x/FT_ALL_CHANGES_2026-01-18_20-00-16/runs/00_FT_ALL_CHANGES_2026-01-18_20-00-16/last/model.pth

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
checkpoint=${CHECKPOINT} \
task.env.objectScaleNoiseMultiplierRange=${objectScaleNoiseMultiplierRange} \
task.env.forceConsecutiveNearGoalSteps=${forceConsecutiveNearGoalSteps} \
task.env.forceScale=${forceScale} \
task.env.torqueScale=${torqueScale} \
task.env.objectAngVelPenaltyScale=${objectAngVelPenaltyScale} \
