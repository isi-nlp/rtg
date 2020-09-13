#!/usr/bin/env bash                                                                                                                                  [37/1983

#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1 --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2
#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
#SBATCH --export=NONE

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = April 3, 2019


RTG_PATH=~/sc2/repos/rtg


# 6 cpus max due to sqlite writes as bottleneck
export RTG_CPUS=4 #$SLURM_CPUS_ON_NODE # 120
export OMP_NUM_THREADS=$RTG_CPUS
export MKL_NUM_THREADS=$RTG_CPUS

#defaults
CONDA_ENV=rtg     # empty means don't activate environment
source ~/.bashrc

usage() {
    echo "Usage: $0  <exp/dir>"
    exit 1;
}

OUT=$1
[[ -n $OUT ]] || usage   # show usage and exit

# rank is the index of current host in list
NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)

echo "Output dir = $OUT"
[[ -d $OUT ]] || mkdir -p $OUT
OUT=$(realpath $OUT)

if [[ ! -f $OUT/rtg.zip ]]; then
    [[ -f $RTG_PATH/rtg/__init__.py ]] || { echo "Error: RTG_PATH=$RTG_PATH is not valid"; exit 2; }
    echo "Zipping source code to $OUT/rtg.zip"
    OLD_DIR=$PWD
    cd ${RTG_PATH}
    zip -r $OUT/rtg.zip rtg -x "*__pycache__*"
    git rev-parse HEAD > $OUT/githead   # git commit message
    cd $OLD_DIR
fi

if [[ -n ${CONDA_ENV} ]]; then
    echo "Activating environment $CONDA_ENV"
    conda activate ${CONDA_ENV} || { echo "Unable to activate $CONDA_ENV" ; exit 3; }
fi


python -m rtg.distrib.launch -P $NUM_GPUS -G 1 -m rtg.pipeline $OUT --fp16

echo "done"
