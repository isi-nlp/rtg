#!/usr/bin/env bash

#$ -P material
#$ -cwd
#$ -pe mt 4
#$ -l h_vmem=4G,h_rt=24:00:00,gpu=1
#$ -l 'h=!vista04&!vista13'

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = April 3, 2019

#SCRIPTS_DIR=$(dirname "${BASH_SOURCE[0]}")  # get the directory name
#RTG_PATH=$(realpath "${SCRIPTS_DIR}/..")


# If using compute grid, and dont rely on this relative path resolution, set the RTG_PATH here
#RTG_PATH=/full/path/to/rtg-master
RTG_PATH=/nas/material/users/tg/work/libs2/rtg-master

OUT=
INP=
OUTP=
BEAMS=4
ALPHA=0.6
CONDA_ENV=torch-3.7     # empty means don't activate environment
ENSEMBLE=5

usage() {
    echo "Usage: $0 -d <exp/dir>
    -i INPUT file to decode
    -o OUTPUT file to store translations
    [-m beam size (default: $BEAMS)]
    [-a length penalty alpha (default: $ALPHA)]
    [-n ensemble models (default: $ENSEMBLE]
    [-e conda_env  default:$CONDA_ENV (empty string disables activation)]
" 1>&2;
    exit 1;
}


while getopts ":d:e:i:o:a:m:n:" o; do
    case "${o}" in
        d) OUT=${OPTARG} ;;
        e) CONDA_ENV=${OPTARG} ;;
        i) INP=${OPTARG} ;;
        o) OUTP=${OPTARG} ;;
        a) ALPHA=${OPTARG} ;;
        m) BEAMS=${OPTARG} ;;
        n) ENSEMBLE=${OPTARG} ;;
        *) usage ;;
    esac
done


[[ -n $OUT ]] || usage   # show usage and exit
[[ -n $INP ]] || usage   # show usage and exit
[[ -n $OUTP ]] || usage   # show usage and exit

#defaults
source ~tg/.bashrc
# TODO: change this -- point to cuda libs
export LD_LIBRARY_PATH=~jonmay/cuda-9.0/lib64:~jonmay/cuda/lib64:/usr/local/lib


#################
#NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)

echo "Output dir = $OUT"
[[ -d $OUT ]] || { echo "$OUT directory not found"; exit 3 ; }

if [[ -n ${CONDA_ENV} ]]; then
    echo "Activating environment $CONDA_ENV"
    source activate ${CONDA_ENV} || { echp "Unable to activate $CONDA_ENV" ; exit 3; }
fi


export PYTHONPATH=$OUT/rtg.zip
echo  "`date`: Starting decoding ... $OUT"

cmd="python -m rtg.decode $OUT -sc -bs $BEAMS -lp $ALPHA -if $INP -of $OUTP -en $ENSEMBLE "
echo "command::: $cmd"
if eval ${cmd}; then
    echo "`date` :: Done"
else
    echo "Error: exit status=$?"
fi
