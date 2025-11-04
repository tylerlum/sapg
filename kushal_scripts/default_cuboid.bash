#!/bin/bash
#SBATCH --partition=humanoid
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodelist=humanoid1
#SBATCH --output=/juno/u/tylerlum/github_repos/sapg/cluster/outputs/%j_rl_train.out
#SBATCH --error=/juno/u/tylerlum/github_repos/sapg/cluster/outputs/%j_rl_train.err

# list out some useful information (optional)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo SLURM_JOB_NAME=$SLURM_JOB_NAME
echo SLURM_JOBID=$SLURM_JOBID
echo SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
echo SLURM_NNODES=$SLURM_NNODES
echo SLURMTMPDIR=$SLURMTMPDIR
echo working directory=$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=$(srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l)
echo NPROCS=$NPROCS
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Get the filename and foldername
# https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOBID' will exist
# `if [ -n $SLURM_JOBID ]` checks if $SLURM_JOBID is not an empty string
if [ -n $SLURM_JOBID ];  then
    # check the original location through scontrol and $SLURM_JOBID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

# Get filename and foldername
export filename=$(basename $SCRIPT_PATH)
export filename_without_ext="${filename%.*}"
export folderpath=$(dirname $SCRIPT_PATH)
export foldername=$(basename $folderpath)

# Use them to set wandb_group, experiment, and hydra_run_dir
export DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
export WANDB_GROUP=$foldername
export EXPERIMENT=00_$filename_without_ext  # Must start with 00_ from code parsing
# export EXPERIMENT=00_$filename_without_ext_$DATETIME  # Add datetime to experiment name to avoid overwriting, end with it so datetime doesn't clog the real name

export HYDRA_RUN_DIR=./train_dir/allegro_kuka_reorientation/$WANDB_GROUP/$EXPERIMENT
# export HYDRA_RUN_DIR=./train_dir/allegro_kuka_reorientation/$WANDB_GROUP/$DATETIME_$EXPERIMENT  # Add datetime to hydra run dir to avoid overwriting, start with it so they are sorted

# Create paths to checkpoints
export CUBOID_PRETRAINED_CHECKPOINT=/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/runs/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/nn/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s.pth

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "filename_without_ext: $filename_without_ext"
echo "foldername: $foldername"
echo "WANDB_GROUP: $WANDB_GROUP"
echo "EXPERIMENT: $EXPERIMENT"
echo "HYDRA_RUN_DIR: $HYDRA_RUN_DIR"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Setup conda environment
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Setting up sapg_env_2"

eval "$(/move/u/tylerlum/miniforge3/bin/conda shell.bash hook)" 
conda activate sapg_env_2

cd /juno/u/tylerlum/github_repos/sapg/

echo "Done setting up sapg_env_2"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Other setup
export WANDB_API_KEY="1d370ac827c148df96f1be99a074adc3398e782b"
export WANDB_DATA_DIR=/move/u/tylerlum/wandb_data
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
export OBJECT_TYPE=cuboid

# Run
command="\
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
wandb_entity=tylerlum \
wandb_activate=True \
wandb_group=$WANDB_GROUP \
wandb_tags='[]' \
++wandb_notes='' \
seed=0 \
experiment=$EXPERIMENT \
hydra.run.dir=$HYDRA_RUN_DIR \
task.env.object_type=$OBJECT_TYPE \
task.env.capture_video=True \
"

echo "Running command: $command"
$command
echo "Done running python script"

# Done
echo "DONE"
