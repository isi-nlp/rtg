#!/usr/bin/env bash
# Created by Thamme Gowda [tg at isi.edu] on July 20, 2019

DIR=`dirname ${BASH_SOURCE[0]}`
DIR=`cd $DIR && pwd` # Full path

log() { echo "$1" >&2; }
usage() {
    echo "${BASH_SOURCE[0]}  -o out/dir :: Directory to store the data
    [ -m ]  :: flag to enable monolingual data download; default=disabled"
}

while getopts ":mo:" op; do
    case "${op}" in
        m) get_mono=true ;;
        o) out_dir=$OPTARG ;;
        \? ) log "Unknown option: -$OPTARG";  usage; exit 1 ;;
        :  ) log "Missing option argument for -$OPTARG"; usage; exit 1 ;;
        *  ) log "Unimplemented option: -$OPTARG"; exit 1 ;;
    esac
done

[[ -n $out_dir ]] || { log "-o <out_dir> is required"; usage; exit 2; }
[[ -d ${out_dir} ]] || mkdir -p ${out_dir}

base_url="http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download"
parallel="parallel.tgz"
dev_test="dev_test.tgz"
mono_hin="monolingual.hi.tgz"

cd $out_dir
out_dir=$PWD # resolve full path

[[ -f _PARALLEL_DONE ]] || {
    wget -c "$base_url/$dev_test"
    tar xvf $dev_test
    wget -c "$base_url/$parallel"
    tar xvf $parallel
    touch _PARALLEL_DONE
}


if [[ -n $get_mono ]]; then
    log "ERROR: Monolingual  download not implemented / completed"
    exit 5
    log "Downloading monolingual data for hindi. This would take some time"
    wget -c $base_url/$mono_hin
    tar xvf $mono_hin

    # english side comes from common crawl
    # wget -c "http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz"
fi

# Setup tokenizer
tools=$DIR/../../tools
[[ -d $tools ]] || mkdir -p $tools
cd $tools

tokr="ulf-tokenizer"
[[ `ls -A $tokr 2> /dev/null` ]] || {
    tokr_url="https://github.com/isi-nlp/ulf-tokenizer/archive/d23b9222"
    wget -c  $tokr_url.tar.gz -O $tokr.tar.gz
    mkdir -p $tokr
    tar xf $tokr.tar.gz -C $tokr --strip-components=1  # remove root level dir name
}

indic_tools='indic_nlp_libray'
[[ `ls -A $indic_tools 2> /dev/null` ]] || {
    indic_url=https://github.com/anoopkunchukuttan/indic_nlp_library/archive/ab3533f
    wget -c  $indic_url.tar.gz -O $indic_tools.tar.gz
    mkdir -p $indic_tools
    tar xf $indic_tools.tar.gz -C $indic_tools --strip-components=1  # remove root level dir name
}

cd $out_dir

tokr=$tools/$tokr
indic_tools=$tools/$indic_tools
src_tokr=$tokr/ulf-src-tok.sh
eng_tokr=$tokr/ulf-eng-tok.sh

indic_norm_tokr() {
    # input > indic_norm | indic_tok | ulf_tok > output
    inp=$1; out=$2; lang=hi
    tmp1=`mktemp`; tmp2=`mktemp`;
    trap "rm $tmp1 $tmp2" EXIT
    export PYTHONPATH=$indic_tools/src
    python -m indicnlp.normalize.indic_normalize $inp $tmp1 hi False
    python -m indicnlp.tokenize.indic_tokenize $tmp1 $tmp2 hi
    $src_tokr < $tmp2 > $out
    rm $tmp1 $tmp2
}

indic_detok() {
    inp=$1; out=$2; lang=hi
    export PYTHONPATH=$indic_tools/src
    python -m indicnlp.tokenize.indic_detokenize $inp $out hi
}

wcl_match() { [[ -f $1 && -f $2 && `wc -l < $1` -eq `wc -l < $2` ]] ; }

for i in {dev_test,parallel}/*.hi; do
    wcl_match $i $i.indtok || { log "$i --> $i.tok"; indic_norm_tokr $i $i.tok ; }
done
for i in {dev_test,parallel}/*.en; do
    wcl_match $i $i.tok || { log "$i --> $i.tok"; $eng_tokr < $i > $i.tok ; }
done

log "All done..."
