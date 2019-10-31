#!/usr/bin/env bash

#$ -P material
#$ -cwd
#$ -pe mt 4
#$ -l h_vmem=8G,h_rt=24:00:00,gpu=1
#$ -l 'h=!vista05'

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = October 17, 2018

SCRIPTS_DIR=$(dirname "${BASH_SOURCE[0]}")  # get the directory name
RTG_PATH=$(realpath "${SCRIPTS_DIR}/..")

# If using compute grid, and dont rely on this relative path resolution, set the RTG_PATH here
#RTG_PATH=/full/path/to/rtg-master

OUT=
CONF_PATH=

#defaults
CONDA_ENV=     # empty means don't activate environment

# TODO: change this -- point to cuda libs
export LD_LIBRARY_PATH=~jonmay/cuda-9.0/lib64:~jonmay/cuda/lib64:/usr/local/lib

usage() {
    echo "Usage: $0 -d <exp/dir>
    [-c conf.yml (default: <exp/dir>/conf.yml) ]
    [-e conda_env  default:$CONDA_ENV (empty string disables activation)] " 1>&2;
    exit 1;
}

while getopts ":fd:c:e:p:" o; do
    case "${o}" in
        d) OUT=${OPTARG} ;;
        c) CONF_PATH=${OPTARG} ;;
        e) CONDA_ENV=${OPTARG} ;;
        *) usage ;;
    esac
done


[[ -n $OUT ]] || usage   # show usage and exit

#################
#NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)

echo "Output dir = $OUT"
[[ -d $OUT ]] || mkdir -p $OUT
OUT=`realpath $OUT`

if [[ ! -f $OUT/rtg.zip || ! -e $OUT/scripts ]]; then
    [[ -f $RTG_PATH/rtg/__init__.py ]] || { echo "Error: RTG_PATH=$RTG_PATH is not valid"; exit 2; }
    echo "Zipping source code to $OUT/rtg.zip"
    OLD_DIR=$PWD
    cd ${RTG_PATH}
    zip -r $OUT/rtg.zip rtg -x "*__pycache__*"
    ln -s ${PWD}/scripts $OUT/scripts  # scripts are needed
    git rev-parse HEAD > $OUT/githead   # git commit message
    cd $OLD_DIR
fi

if [[ -n ${CONDA_ENV} ]]; then
    echo "Activating environment $CONDA_ENV"
    source activate ${CONDA_ENV} || { echo "Unable to activate $CONDA_ENV" ; exit 3; }
fi


export PYTHONPATH=$OUT/rtg.zip
# copy this script for reproducibility
cp "${BASH_SOURCE[0]}"  $OUT/job.sh.bak
echo  "`date`: Starting pipeline... $OUT"

CONF_ARG="$CONF_PATH"
if [[ -f $OUT/conf.yml && -n $CONF_PATH ]]; then
    echo "ignoring $CONF_PATH, because $OUT/conf.yml exists"
    CONF_ARG=""
fi

cmd="python -m rtg.pipeline $OUT $CONF_ARG --gpu-only"
echo "command::: $cmd"
if eval ${cmd}; then
    echo "`date` :: Done"
else
    echo "Error: exit status=$?"
fi
