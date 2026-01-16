#!/usr/bin/env zsh
source ~/.zshrc
# export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer_2/clockwise
# export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer_2/counter_clockwise
# export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/mallet/clockwise
export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/mallet/counter_clockwise

sam2_ros_env

python video_sam2.py \
--input_dir $DEMO_DIR/rgb/ \
--output_dir $DEMO_DIR/hand_mask/ \
--use_negative_prompt

hamer_depth_env

python run.py \
--rgb-path $DEMO_DIR/rgb \
--depth-path $DEMO_DIR/depth \
--mask-path $DEMO_DIR/hand_mask \
--cam-intrinsics-path $DEMO_DIR/cam_K.txt \
--out-path $DEMO_DIR/hand_pose_trajectory \
--hand-type LEFT