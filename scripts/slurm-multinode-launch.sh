#!/usr/bin/env bash

#SBATCH --mem=24G 
#SBATCH --time=0-24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:2
## SBATCH --mem-per-gpu=10G

#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
## SBATCH --export=NONE

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date created = April 3, 2019
#  Last revised: August 22, 2020

#SCRIPTS_DIR=$(dirname "${BASH_SOURCE[0]}")  # get the directory name
#RTG_PATH=$(realpath "${SCRIPTS_DIR}/..")
RTG_PATH=~/sc2/repos/rtg

source ~/.bashrc 

OUT=
JOB_SCRIPT=slurm-multinode-job.sh

usage() {
    echo "Usage: $0 -d <exp/dir>
    [-u update the code, get the latest from $RTG_PATH]" 1>&2;
    exit 1;
}


while getopts ":ud:c:e:p:r::f" o; do
    case "${o}" in
        d) OUT=${OPTARG} ;;
        u) UPDATE=YES ;;
        *) usage ;;
    esac
done


[[ -n $OUT ]] || usage   # show usage and exit
[[ -d $OUT ]] || mkdir -p $OUT
[[ -f $JOB_SCRIPT ]] || { echo "Job script $JOB_SCRIPT not found"; exit 2; }

echo "Output dir = $OUT"
OUT=$(realpath $OUT)

if [[ ! -f $OUT/rtg.zip  || -n $UPDATE ]]; then
    [[ -f $RTG_PATH/rtg/__init__.py ]] || { echo "Error: RTG_PATH=$RTG_PATH is not valid"; exit 2; }
    echo "Zipping source code from $RTG_PATH to $OUT/rtg.zip"
    OLD_DIR=$PWD
    cd ${RTG_PATH}

    zip -r $OUT/rtg.zip rtg -x "*__pycache__*"
    git rev-parse HEAD > $OUT/githead   # git commit message
    cd $OLD_DIR
fi

# copy this script for reproducibility
[[ -f $OUT/job-launch.sh.bak ]] || cp "${BASH_SOURCE[0]}" $OUT/job-launch.sh.bak
[[ -f $OUT/job-job.sh.bak ]] || cp $JOB_SCRIPT $OUT/job-job.sh.bak

echo "Nodes: $(scontrol show hostnames)"

# run jobs
srun $JOB_SCRIPT -d $OUT
