#!/usr/bin/env bash

# Created by Thamme Gowda on June 27, 2019
DIR=`dirname ${BASH_SOURCE[0]}`
DIR=`cd $DIR && pwd` # Full path

tools=$DIR/../tools  # where the tokenizer should be stored
[[ -d $tools ]] || mkdir -p $tools


inp=/dev/stdin
out=/dev/stdout
lang=hi
detok=

log() { >&2 echo "$1"; }

usage() {
    >&2 echo "${BASH_SOURCE[0]}
    [-i input_file (default=$inp) ]
    [-o output_file (default=$out)]
    [-l language (default=$lang)]
    [-d :: flag for detokenization mode. default; this flag is unset, so tokenize mode]"
}

while getopts ":di:o:l:" op; do
    case "${op}" in
        d) detok=true ;;
        i) inp=$OPTARG ;;
        o) inp=$OPTARG ;;
        l) lang=$OPTARG ;;
       \? ) log "Unknown option: -$OPTARG";  usage; exit 1 ;;
        : ) log "Missing option argument for -$OPTARG"; usage; exit 1 ;;
        * ) log "Unimplemented option: -$OPTARG"; exit 1 ;;
    esac
done


indic_tools="$tools/indic_nlp_libray"
ls -A $indic_tools > /dev/null 2>&1  || {
    log "Downloading indic_nlp_library to $indic_tools"
    indic_url=https://github.com/anoopkunchukuttan/indic_nlp_library/archive/ab3533f
    wget -c $indic_url.tar.gz -O $indic_tools.tar.gz
    mkdir -p $indic_tools
    tar xf $indic_tools.tar.gz -C $indic_tools --strip-components=1  # remove root level dir name
}

indic_norm_tokr() {
    # input > indic_norm | indic_tok > output
    log "Norm+tokenizing ... $lang: $inp -> $out"
    tmp1=`mktemp`
    trap "rm $tmp1 " EXIT
    export PYTHONPATH=$indic_tools/src
    python -m indicnlp.normalize.indic_normalize $inp $tmp1 $lang False
    python -m indicnlp.tokenize.indic_tokenize $tmp1 $out $lang
    #rm $tmp1
}

indic_detok() {
    log "Detokenizing ... $lang: $inp -> $out"
    export PYTHONPATH=$indic_tools/src
    python -m indicnlp.tokenize.indic_detokenize $inp $out $lang
}

[[ -n $detok ]] && indic_detok || indic_norm_tokr
log "Done"