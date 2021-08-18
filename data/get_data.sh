#!/usr/bin/env bash
set -e # exit on error

DIR="$(dirname "${BASH_SOURCE[0]}")"  # Get the directory name
#DIR="$(realpath "${DIR}")"    # Resolve its full path if need be

log_exit() { echo "$2"; exit 1; }

# note, use sacremoses 0.0.45 or newer;
# I contributed nukthas and viramas for indic langs

for cmd in cut sed unzip sacremoses mtdata awkg ; do
  which $cmd &> /dev/null ||
    log_exit 1 "$cmd not found; please install $cmd and rerun me."
done


function tokenize {
    raw=$1
    tok=$2
    echo "tokenizing $raw --> $tok"
    [[ -f $raw ]] || log_exit 2 "input file not found $raw"
    #[[ -f $tok ]] && log_exit 2 "output file is not empty $tok"
    cat $raw | html_unescape | sacremoses normalize -q -d -p -c tokenize -a -x -p :web: > $tok
 }

function html_unescape {
    sed -E 's/\& (ge|le|gt|lt|amp|quot|apos|nbsp);/\&\1;/g' |
        awkg -b 'from html import unescape' 'print(unescape(R0))'
}

function get_hin_eng {

    dest="$1"
    [[ -e $dest/_GOOD ]] && return
    [[ -d $dest ]] || mkdir -p $dest

    [[ -f $dest/mtdata.signature.txt ]] || {
        mtdata get --langs hin-eng --merge --out $dest \
               --train IITBv1_5_train --test IITBv1_5_{dev,test}

        mv $dest/train.hin{,.bak}
        mv $dest/train.eng{,.bak}
        # grep -E '^https?:[^ ]*$'
        # exclude copy
        paste $dest/train.{hin,eng}.bak | awkg -F '\t' 'RET=R[0] != R[1]' > $dest/train.hin-eng       
        cut -f1 $dest/train.hin-eng > $dest/train.hin
        cut -f2 $dest/train.hin-eng > $dest/train.eng
     }

    for lang in eng hin; do
        for split in dev test; do
            link=$dest/$split.$lang
            [[ -e $link ]] ||
                ln -s tests/IITBv1_5_$split-hin_eng.$lang $link
        done
    done
    
    for split in dev test train; do
        for lang in eng hin; do
            tok_file=$dest/$split.$lang.tok
            [[ -s $tok_file ]] || tokenize $dest/$split.$lang $tok_file
        done
    done
    touch $dest/_GOOD
}


function get_deu_eng {
    dest="$1"
    [[ -e $dest/_GOOD ]] && return
    [[ -d $dest ]] || mkdir -p $dest

    [[ -f $dest/mtdata.signature.txt ]] || {
       mtdata get --langs deu-eng --merge --out $dest \
           --train news_commentary_v14 --test newstest201{8,9}_deen

        mv $dest/train.deu{,.bak}
        mv $dest/train.eng{,.bak}
        # grep -E '^https?:[^ ]*$'
        # exclude copy
        paste $dest/train.{deu,eng}.bak | awkg -F '\t' 'RET=R[0] != R[1]' > $dest/train.deu-eng
        cut -f1 $dest/train.deu-eng > $dest/train.deu
        cut -f2 $dest/train.deu-eng > $dest/train.eng
     }

    for lang in eng deu; do
      printf "dev newstest2018_deen-deu_eng\ntest newstest2019_deen-deu_eng\n" | \
        while read split name; do
          link=$dest/$split.$lang
          [[ -e $link ]] || ln -s tests/$name.$lang $link
        done
     done
    for split in dev test train; do
        for lang in eng deu; do
            tok_file=$dest/$split.$lang.tok
            [[ -s $tok_file ]] || tokenize $dest/$split.$lang $tok_file
        done
    done
    touch $dest/_GOOD
}


get_hin_eng hin-eng
get_deu_eng deu-eng








