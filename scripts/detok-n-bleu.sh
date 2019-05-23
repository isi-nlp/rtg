#!/usr/bin/env bash

DIR=$(dirname "${BASH_SOURCE[0]}")  # get the directory name
DIR=$(realpath "${DIR}")    # resolve its full path if need be

usage() {
    echo "Usage: $0 -h hypotheses -r references" 1>&2;
    exit 1;
}

HYP=
REF=

while getopts ":h:r:" o; do
    case "${o}" in
        h)
            HYP=${OPTARG}
            ;;
        r)
            REF=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n ${REF}  || -n ${HYP} ]] || usage


# detokenize first column
cut -f1 ${HYP} | ${DIR}/detokenizer.perl | sed 's/ @\([^@]\+\)@ /\1/g' > ${HYP}.detok

# multi bleu
${DIR}/multi-bleu-detok.perl ${REF} < ${HYP}.detok > ${HYP}.detok.tc.bleu
${DIR}/multi-bleu-detok.perl -lc ${REF} < ${HYP}.detok > ${HYP}.detok.lc.bleu