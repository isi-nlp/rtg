#!/usr/bin/env bash
#
#  This script downloads and prepares parallel data from WMT19,
#  as of now, europarl v9 and europarl v7 parallel data downloading is supported
#
#  Author : Thamme Gowda [tg@isi.edu]
#  Created: Oct 15, 2019
#
#
set -e   # Exit immediately if a command returns a non-zero status.

log(){
    printf "${@}\n" >&2
}

maybe_mkdir(){
    #args: <d1> [<d2> ...]
    local dir
    for dir in "$@"; do
        [[ -d "$dir"  ]] || {
            log "Creating $1"
            mkdir -p "$dir"
        }
    done
}

maybe_download(){
    #args: <local_file> <URL>
    local file="$1"
    local url="$2"
    local flag="${file}.DOWNLOADED"
    [[ -f "$flag" ]] && { log "Skip: $url --> $file"; return; }
    log "Download: $url -> $file"
    wget --continue -O "$file" "$url"
    touch $flag
}

download_all(){
    local dir="$1"
    local -n files=$2           # name ref; like pointer to some array
    euro_parl_v9="http://www.statmt.org/europarl/v9/training"
    de_en="$euro_parl_v9/europarl-v9.de-en.tsv.gz"
    fi_en="$euro_parl_v9/europarl-v9.fi-en.tsv.gz"
    lt_en="$euro_parl_v9/europarl-v9.lt-en.tsv.gz"
     # cant find these in euro_parl_v9
     euro_parl_v7="http://www.statmt.org/europarl/v7"
    fr_en="$euro_parl_v7/fr-en.tgz"
    bg_en="$euro_parl_v7/bg-en.tgz"
    wmt19_dev="http://data.statmt.org/wmt19/translation-task/dev.tgz"

    queue=($de_en $fi_en $lt_en $fr_en $bg_en $wmt19_dev)
    local url
    for url in "${queue[@]}"; do
        file=$dir/$(basename $url)
        maybe_download "$file" "$url"
        files+=($file)
     done
}


extract_all(){
    # <data_dir> <in_file_arr> <out_file_arr>
    local data_dir=$1
    local -n in_files=$2
    local -n out_dirs=$3
    local file
    for file in "${in_files[@]}"; do
        if [[ "$file" == *.tgz ]]; then
            # XX-YY.tgz  or dev.tgz
            pair=$(basename $file | cut -d. -f1)
            out_dir="${data_dir}/$pair"
            maybe_mkdir "$out_dir"
            [[ "$(basename $file)" == "dev.tgz" ]] && local opts="--strip=1" || local opts=

            if [[ -f $out_dir/_EXTRACTED ]]; then
                log "Skip: $out_dir"
            else
                log "tar xvf $file $opts -C $out_dir "
                tar xvf "$file" $opts -C "$out_dir"  && touch $out_dir/_EXTRACTED
            fi

        elif [[ $file == *.tsv.gz ]]; then
            # europarl-v9.XX-YY.tsv.gz
            pair=$(basename $file | cut -d. -f2)
            out_dir="${data_dir}/$pair"
            maybe_mkdir $out_dir
            if [[ -f $out_dir/_EXTRACTED ]]; then
                log "Skip: $out_dir"
            else
                out_file=$(echo $out_dir/$(basename $file)| sed 's/.gz$//')
                log "gunzip -ck $file > $out_file"
                gunzip -ck $file > $out_file
                first=$(echo $out_file | sed 's/.\([[:alpha:]]*\)-\([[:alpha:]]*\).tsv/.\1-\2.\1/')
                second=$(echo $out_file | sed 's/.\([[:alpha:]]*\)-\([[:alpha:]]*\).tsv/.\1-\2.\2/')
                [[ "$first" = "$second" || "$first" == "$out_file" ]] && { log "Error: $out_file --> $first $second"; exit 5; }
                cut -f1 $out_file > $first
                cut -f2 $out_file > $second
                touch $out_dir/_EXTRACTED
            fi
        else
            log "Error: Dont know how do handle $file "
            exit 4
        fi
        out_dirs+=("$out_dir")
    done
}

get_tokenizer(){
    moses_code_url="https://github.com/moses-smt/mosesdecoder/archive/master.tar.gz" # tested on oct 15, 2019
    moses_code_zip="${DOWNLD}/mosesdecoder.tar.gz"
    moses_code=${TOOLS}/mosesdecoder
    maybe_download ${moses_code_zip} ${moses_code_url}
    tokenizer="${moses_code}/scripts/tokenizer/tokenizer.perl"
    if [[ ! -f "$tokenizer" ]]; then
        log "extract $moses_code_zip -> $moses_code"
        maybe_mkdir "$moses_code"
        tar xf "$moses_code_zip" -C "$moses_code" --strip=1  >&2
    fi
    [[ -f "$tokenizer" ]] && echo $tokenizer || { log  "Error: Couldnt setup Moses tokr"; exit 6; }
}

tokenize_all(){
    local -n dirs=$1
    local tokr=$(get_tokenizer)
    local dir
    threads=${THREADS:-$(nproc)}
    log "tokenizer = $tokr; threads=$threads"
    for dir in "${dirs[@]}"; do
        log "Tokenizing $dir"
        local file
        for file in $(find $dir -type f \( ! -name "*.sgm" -a ! -name "*.tsv" -a ! -name "_*" -a ! -name ".*" \)); do
            lang=${file##*.}
            if [[ "$lang" && " $LANGS " =~ " $lang " ]]; then
                file_tok="$file.tok"
                if [[ -s $file_tok && $(wc -l < $file) -eq $(wc -l < $file_tok) ]]; then
                    log "Tok Skip : $file"
                else
                    cmd="$tokr -l $lang -no-escape -threads $threads < $file > $file.tok"
                    log "$cmd"
                    eval "$cmd"
                fi
            fi
        done
    done
}

usage(){
    log "Usage:\n get-wmt-data.sh --work <WORK_DIR>\n\n\
    <WORK_DIR> is a directory to be used for downloading data and tools\n" >&2
    exit 1
}

main(){
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
          --work)
            WORK="$2"; shift 2;;
          *)
            log "ERROR: $key argument unknown"
            usage ;;
        esac
    done

    [[ -n "$WORK" ]] || { log "ERROR: --work directory path not provided"; exit 2; }
    [[ -d "$WORK" ]] || { log  "Creating dir: $WORK"; maybe_mkdir "$WORK"; }

    export WORK=$(cd "$WORK" && pwd )
    export DATA=$WORK/data
    export DOWNLD=$WORK/downloads
    export TOOLS=$WORK/tools
    export LANGS="en de fr fi lt bg" # languages to be focused on; exclude others from tokenization
    maybe_mkdir $DATA $DOWNLD $TOOLS
    local dl_files=()
    download_all $DOWNLD dl_files
    local lang_dirs=()

    extract_all "$DATA" dl_files lang_dirs
    tokenize_all lang_dirs

}

main "${@}"