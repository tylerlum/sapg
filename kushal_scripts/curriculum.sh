#!/bin/bash

# experiment_name="domain_randomization_slow_no_extra_obs_single_gpu_pretraining"
experiment_name="newFinish5_finetuneScannedHammer_RelativeControl_10_movingAverage_1_dropout"

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
wandb_group=pretrain \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=00_${experiment_name} \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/${experiment_name} \
task.env.capture_video=True \
task.env.object_type=scanned_hammer_1 \
task.env.asset.kukaAllegro=urdf/kuka_allegro_description/iiwa14_real.urdf \
checkpoint=/share/portal/kk837/sapg/train_dir/allegro_kuka_reorientation/relativeRuns_10/runs/00_relativeRuns_10/last/model.pth \
task.env.actionsMovingAverage=1.0 \
task.env.dofSpeedScale=10 \
task.env.useRelativeControl=True \
task.env.turn_off_extra_obs_slowly=True \
task.env.turn_off_palm_vel_obs_slowly=True \
task.env.turn_off_object_vel_obs_slowly=True \
task.env.dofSpeedScaleFinal=5 \
task.env.actionsMovingAverageFinal=0.1 \
task.env.use_obs_dropout=True \
# task.task.randomize=True \
# task.env.turn_off_extra_obs=True \
# task.env.turn_off_object_vel_obs=True \

