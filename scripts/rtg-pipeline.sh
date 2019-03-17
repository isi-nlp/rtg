#!/usr/bin/env bash

#$ -P material
#$ -cwd
#$ -pe mt 6
#$ -l h_vmem=8G,h_rt=24:00:00,gpu=1
#$ -l 'h=!vista05'

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = October 17, 2018

OUT=
CONF_PATH=
BATCH_SIZE=56

usage() {
    echo "Usage: $0 -d <exp/dir>
    [-c conf.yml] " 1>&2;
    exit 1;
}

while getopts ":fd:c:" o; do
    case "${o}" in
        d)
            OUT=${OPTARG}
            ;;
        c)
            CONF_PATH=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n $OUT ]] || usage

#################

# These needs to be changed
source ~tg/.bashrc
source activate torch-3.7
#export PYTHONPATH=.:~tg/work/libs2/rtg-master
export LD_LIBRARY_PATH=~jonmay/cuda-9.0/lib64:~jonmay/cuda/lib64:/usr/local/lib
#NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
####

echo "Output dir = $OUT"
[[ -d $OUT ]] || mkdir -p $OUT

if [[ ! -f $OUT/rtg.zip || ! -e $OUT/scripts ]]; then
    echo "Zipping source code to $OUT/rtg.zip"
    OLD_DIR=$PWD
    cd ~tg/work/libs2/rtg-master
    zip -r $OUT/rtg.zip rtg -x "*__pycache__*"
    ln -s ${PWD}/scripts $OUT/scripts  # scripts are needed
    git rev-parse HEAD > $OUT/githead   # git commit message
    cd $OLD_DIR
fi

export PYTHONPATH=$OUT/rtg.zip
# copy this script for reproducibility
cp "${BASH_SOURCE[0]}"  $OUT/job.sh.bak
echo  "Starting pipeline... $OUT"

[[ -n $CONF_PATH ]] && C="$CONF_PATH"

if [[ -f $OUT/conf.yml && -n $CONF_PATH ]]; then
    echo "ignoring $CONF_PATH, because $OUT/conf.yml exists"
    CONF_ARG=""
else
    CONF_ARG="$CONF_PATH"
fi

cmd="python -m rtg.pipeline $OUT $CONF_ARG"
echo "command::: $cmd"
if eval ${cmd}; then
    echo "Done"
else
    echo "Error: exit status=$?"
fi