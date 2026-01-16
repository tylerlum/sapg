#!/usr/bin/env zsh
source ~/.zshrc

export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer_2/clockwise
# export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer_2/counter_clockwise
# export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/mallet/clockwise
# export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_15/mallet/counter_clockwise

export OBJECT_PATH=assets/urdf/dex_tool_bench/hammer/hammer_2/hammer_2.urdf

sapg

python human_demo/visualize_demo.py \
--object-path $OBJECT_PATH \
--object-poses-json-path $DEMO_DIR/subgoals.json \
--hand-poses-dir $DEMO_DIR/hand_pose_trajectory/ \
--visualize-hand-meshes