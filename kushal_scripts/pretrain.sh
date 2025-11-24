#!/bin/bash

# experiment_name="domain_randomization_slow_no_extra_obs_single_gpu_pretraining"
experiment_name="DR_RELATIVE_SLOW"

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
wandb_project=pretraining_ablations \
wandb_entity=kk837 \
wandb_activate=True \
wandb_group=${experiment_name} \
wandb_tags=[] \
++wandb_notes='' \
seed=0 \
experiment=00_${experiment_name} \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/${experiment_name} \
task.env.object_type=cuboid \
task.env.capture_video=True \
task.env.handMovingAverage=0.1 \
task.env.armMovingAverage=0.1 \
task.env.dofSpeedScale=1 \
task.env.useRelativeControl=True \
task.env.asset.kukaAllegro=urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted.urdf \
task.task.randomize=True \
# task.env.asset.kukaAllegro=urdf/kuka_allegro_description/iiwa14_real.urdf \

