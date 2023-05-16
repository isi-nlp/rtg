#!/usr/bin/env bash
# Author= Thamme Gowda
# Created = June 2021

DIR="$(dirname "${BASH_SOURCE[0]}")"  # Get the directory name
#DIR="$(realpath "${DIR}")"    # Resolve its full path if need be
DIR="$(cd $DIR/../.. && pwd)"

set -e # exit on error

log_exit() { echo "$2"; exit 1; }

for cmd in wget cut sed unzip sacremoses; do
  which $cmd &> /dev/null ||
    log_exit 1 "$cmd not found; please install $cmd and rerun me."
done

lowercase() {
     python -c "
import sys, io
stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
for line in stdin:
  print(line.strip().lower())
"
}
function get_trec {
    dest="$1"
    [[ -e $dest/_GOOD ]] && return
    [[ -d $dest ]] || mkdir -p $dest

    TRAIN="https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
    TEST="https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
    trec_5500=$dest/TREC_5500.orig
    test=$dest/TREC_10.orig
    [[ -s $trec_5500 ]] || wget "$TRAIN" -O "$trec_5500"
    [[ -s $test ]] || wget "$TEST" -O "$test"

    # create validation set (10% of original training set)
    train=$dest/train.orig
    valid=$dest/valid.orig
    [[ -s $trec_5500.shuf ]] || shuf < $trec_5500 > $trec_5500.shuf
    [[ -s $valid ]] || awk "NR <= 400" < $trec_5500.shuf > $valid
    [[ -s $train ]] || awk "NR > 400" < $trec_5500.shuf > $train

    echo "$dest/train $train
         $dest/valid $valid
         $dest/test $test" |
        while read pref file; do
            cat $file | cut -f1 -d' ' | cut -f1 -d: > $pref.coarse
            cat $file | cut -f1 -d' ' | cut -f2 -d: > $pref.fine
            cat $file | sed 's/^[^ ]* //' > $pref.text
            lowercase < $pref.text > $pref.text.lc
        done
    echo "TREC datset is ready at $dest"
    touch $dest/_GOOD
}

get_trec $DIR/data/trec