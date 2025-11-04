#!/bin/bash

# experiment_name="correct_slowed_turn_off_obs_randomize_multigpu_pretraining"
experiment_name="reduced_batch_size_vanilla_multigpu_pretraining"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
train.py \
task=AllegroKukaLSTM \
task/env=reorientation \
++task.env.useSparseReward=False \
headless=True \
task.env.numEnvs=8192 \
train.params.config.minibatch_size=32768 \
multi_gpu=True \
train.params.config.good_reset_boundary=0 \
task.env.goodResetBoundary=0 \
train.params.config.use_others_experience=lf \
train.params.config.off_policy_ratio=1.0 \
train.params.config.expl_type=mixed_expl_learn_param \
train.params.config.expl_reward_type=entropy \
train.params.config.expl_coef_block_size=2048 \
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
task.env.object_type=cuboid \
# task.env.actionsMovingAverage=0.1 \
# task.env.dofSpeedScale=1.0 \
# task.env.turn_off_extra_obs=True \
# task.env.turn_off_object_vel_obs=True \
# task.task.randomize=True \
