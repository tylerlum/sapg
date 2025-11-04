#!/bin/bash

object_type='YcbHammer'
experiment_name="hammer_scratch_all_goals"

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
wandb_group=scratch_train \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=00_${experiment_name} \
hydra.run.dir=./train_dir/allegro_kuka_reorientation/${experiment_name} \
task.env.object_type=${object_type} \
task.env.use_fixed_set_of_goal_states=False \
task.env.use_fixed_init_object_pose=False \
task.env.envSpacing=1.0 \
task.env.maxConsecutiveSuccesses=50 \
task.env.capture_video=True

# checkpoint=/share/portal/kk837/sapg/train_dir/checkpoints/cuboid_pretrain_tyler/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s.pth \