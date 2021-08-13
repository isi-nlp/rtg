#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/15/21
from rtg.emb import tfmcls
from rtg.registry import log
import subprocess

from torchtext.datasets import DBpedia
from pathlib import Path
import random
import rtg
from rtg.registry import registry, MODEL
import pytest
from rtg.pipeline import Pipeline, Experiment
import tempfile
from rtg.exp import load_conf
import torch
import shutil
from . import sanity_check_experiment


def copy_head(inp: Path, out: Path, head: int):
    count = 0
    with inp.open('r') as inp, out.open('w') as out:
        for line in inp:
            out.write(line.strip() + '\n')
            count += 1
            if count >= head:
                break


def setup_dataset():
    root = rtg.RTG_PATH / '.data'
    data_dir = Path(__file__).parent / 'test-data'
    dbpedia_dir = data_dir / 'dbpedia'
    flag = dbpedia_dir / '_VALID'

    if not flag.exists():
        dbpedia_dir.mkdir(exist_ok=True, parents=True)
        test = list(DBpedia(root=root, split='test'))
        train = list(DBpedia(root=root, split='train'))
        random.shuffle(train)
        train = train[:10_000]  # for quick testing
        ten_per = int(0.1 * len(train))
        valid, train = train[:ten_per], train[ten_per:]

        fargs = dict(mode='w', encoding='utf8', errors='ignore')
        for name, data, head in [('train', train, 10_000), ('valid', valid, 1_000),
                                 ('test', test, None)]:
            log.info(f"Writing {name}")
            text_f = dbpedia_dir / f'{name}.text'
            label_f = dbpedia_dir / f'{name}.label'
            with text_f.open(**fargs) as text, label_f.open(**fargs) as label:
                for l, t in data:
                    text.write(t.replace("\n", " ").replace("\t", " ").strip() + '\n')
                    label.write(f'{l}\n')
        flag.touch()


setup_dataset()


def test_tfmcls_model():
    # tmp_dir = tempfile.mkdtemp()
    tmp_dir = Path('tmp.dbpedia-exp')
    config = load_conf('experiments/transformer.classifier.yml')
    exp = registry[MODEL]['tfmcls'].experiment(tmp_dir, config=config, read_only=False)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    # exp.config['prep']['num_samples'] = 0
    Pipeline(exp).run(run_tests=False)
    sanity_check_experiment(exp, samples=False, shared_vocab=False)
    print(f"Cleaning up {tmp_dir}")
    # shutil.rmtree(tmp_dir, ignore_errors=True)



