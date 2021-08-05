#!/usr/bin/env bash
# use this script to synchronize private repo with public repo

public_remote=git@github.com:isi-nlp/rtg.git

git push $public_remote public:master
git pull $public_remote master:public