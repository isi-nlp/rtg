#!/usr/bin/env bash
# Author: Thamme Gowda
# Created: Oct 2021

DOCS_DIR="$(dirname "${BASH_SOURCE[0]}")"  # Get the directory name
DOCS_DIR="$(realpath "${DOCS_DIR}")"            # Resolve its full path if need be

function my_exit {
  # my_exit <exit-code> <error-msg>
  echo "$2" 1>&2
  exit "$1"
}

rtg_version=$(python -m rtg.pipeline -v 2> /dev/null | cut -d' ' -f2 | sed 's/-.*//')   # sed out  -dev -a -b etc
[[ -n $rtg_version ]] || my_exit 1 "Unable to parse rtg version; check: python -m rtg.pipeline -v"
echo "rtg $rtg_version"
asciidoctor -v || my_exit 2 "Asciidoctor not found; please install and rerun"

ver_dir="${DOCS_DIR}/v${rtg_version}"
[[ -d $ver_dir ]] || mkdir -p "$ver_dir"
cmd="asciidoctor -o ${ver_dir}/index.html $DOCS_DIR/index.adoc"
echo "Running:: $cmd"
eval "$cmd" || my_exit 3 "Doc building Failed"

if [[ -f "$DOCS_DIR/index.html" ]]; then
  rm "$DOCS_DIR/index.html"
  ln -s "v$rtg_version/index.html" "$DOCS_DIR/index.html"
fi

[[ -f $DOCS_DIR/versions.adoc ]] &&  asciidoctor -o "$DOCS_DIR/versions.html" "$DOCS_DIR/versions.adoc"
