#!/usr/bin/env bash

#$ -P material
#$ -cwd
#$ -pe mt 6
#$ -l h_vmem=8G,h_rt=24:00:00,gpu=1
#$ -l 'h=!vista04'

# Pipeline script for MT
#
# Author = Thamme Gowda (tg@isi.edu)
# Date = October 17, 2018

OUT=
CONF_PATH=
BATCH_SIZE=56
STEPS=128000
KEEP=20
BEAM_SIZE=4

usage() {
    echo "Usage: $0 -d <exp/dir>
        [-c conf.yml]
        [-b batch_size (default:$BATCH_SIZE)]
        [-s steps (default:$STEPS)]
        [-k keep_models (default:$KEEP)]
        [-m beam_size (default:$BEAM_SIZE)]" 1>&2;
    exit 1;
}

while getopts ":d:c:b:e:k:m:" o; do
    case "${o}" in
        d)
            OUT=${OPTARG}
            ;;
        c)
            CONF_PATH=${OPTARG}
            ;;
        b)
            BATCH_SIZE=${OPTARG}
            ;;
        k)
            KEEP=${OPTARG}
            ;;
        s)
            STEPS=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n $OUT ]] || usage

EXP_DIR=$OUT
#################

# These needs to be changed
source ~tg/.bashrc
source activate torch
export PYTHONPATH=.:~tg/work/libs2/rtg

export LD_LIBRARY_PATH=~jonmay/cuda-9.0/lib64:~jonmay/cuda/lib64:/usr/local/lib
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

####

echo "Output dir = $OUT"
[[ -d $OUT ]] || mkdir -p $OUT

function log {
    printf "`date '+%Y-%m-%d %H:%M:%S'`:: $1\n" >> $OUT/job.log
}

# copy this script for reproducibility
cp "${BASH_SOURCE[0]}"  $OUT/job.sh.bak
log "Starting... \nDescription=$DESC \n"

########### PREPARE #########
if [[ ! -f $OUT/_PREPARED ]]; then
    # if conf is not provided but there is one inside the directory, use it
    [[ -z $CONF_PATH && -f $OUT/conf.yml ]] && CONF_PATH=$OUT/conf.yml

    [[ -n $CONF_PATH ]] || { echo "conf_path is needed"; usage; }
    [[ -f $CONF_PATH ]] || { echo "File conf $CONF_PATH doesnt exist"; exit 4; }

    log "Step : preparing experiment. $CONF_FILE"
    cmd="python -m rtg.prep $EXP_DIR $CONF_PATH "
    log "$cmd"
    if ! eval "$cmd"; then
        log 'Failed... exiting'
        exit 1
    fi
fi


####### TRAIN ########
if [[ ! -f "$OUT/_TRAINED" ]]; then
    log "Step : Starting trainer"
    cmd="python -m rtg.train $EXP_DIR --steps $STEPS --keep-models $KEEP --batch-size $BATCH_SIZE"
    log "$cmd"
    if eval "$cmd"; then
        touch $OUT/_TRAINED
    else
        log 'Failed....'
        exit 2
    fi
fi


########## DECODE and SCORE #####
function decode {
    # accepts two args: <src-file> <out-file>
    FROM=$1
    TO=$2
    cmd="python -m rtg.decode $EXP_DIR --beam-size $BEAM_SIZE --input $FROM --output $TO"
    log "$cmd"
    eval "$cmd"
}

function score {
    # usage: score <output> <reference>
    out=$1
    ref=$2
    DETOK=~tg/bin/detokenize.perl
    BLEU=~tg/bin/multi-bleu-detok.perl

    cut -f1 $out | $DETOK | sed 's/ @\([^@]\+\)@ /\1/g' > $out.detok
    cat $out.detok | $BLEU  $ref > $out.detok.tc.bleu
    cat $out.detok | $BLEU -lc $ref > $out.detok.lc.bleu
}


######## Test data files #########
DATA=/nas/material/users/tg/work/data/material/y2/merged/1S-buildldc
SRC='src.tok'
REF='ref'

DEV=$DATA/1S-builddev
TEST=$DATA/1S-buildtest
LDCDEV=$DATA/elisa-som-dev
LDCTEST=$DATA/elisa-som-test

test_dir=$(printf "$OUT/test_%03d_%03d" $STEPS $BEAM_SIZE)
mkdir -p $test_dir

printf "$TEST test
$DEV dev
$LDCDEV ldcdev
$LDCTEST ldctest
" | while read pref split; do
    echo "Decoding and scoring $split Source:$pref.$SRC Ref: $pref.$REF"
    ln -s $pref.$SRC $test_dir/$split.src
    ln -s $pref.$REF $test_dir/$split.ref

    decode $test_dir/$split.src $test_dir/$split.out
    score $test_dir/$split.out $test_dir/$split.ref
done

log "Done"
