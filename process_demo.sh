#!/usr/bin/env zsh
set -e  # exit on error
source ~/.zshrc

# -------------------------------
# Argument handling
# -------------------------------
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <DEMO_DIR>"
  exit 1
fi

DEMO_DIR="$1"

if [[ ! -d "$DEMO_DIR" ]]; then
  echo "Error: DEMO_DIR does not exist: $DEMO_DIR"
  exit 1
fi

echo "Processing demo: $DEMO_DIR"

# -------------------------------
# SAM2 hand mask
# -------------------------------
sam2_ros_env

python video_sam2.py \
  --input_dir "$DEMO_DIR/rgb/" \
  --output_dir "$DEMO_DIR/hand_mask/" \
  --use_negative_prompt

# -------------------------------
# HAMER depth / hand pose
# -------------------------------
hamer_depth_env

python run.py \
  --rgb-path "$DEMO_DIR/rgb" \
  --depth-path "$DEMO_DIR/depth" \
  --mask-path "$DEMO_DIR/hand_mask" \
  --cam-intrinsics-path "$DEMO_DIR/cam_K.txt" \
  --out-path "$DEMO_DIR/hand_pose_trajectory" \
  --hand-type LEFT
