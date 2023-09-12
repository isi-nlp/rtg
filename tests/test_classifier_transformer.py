#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 6/15/21
import random
from pathlib import Path
import tempfile
import shutil


from rtg import load_conf, log, registry, RTG_PATH, MODELS

from rtg.cli.pipeline import Pipeline
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
    root = RTG_PATH / '.data'
    data_dir = Path(__file__).parent / 'data'
    dbpedia_dir = data_dir / 'dbpedia'
    flag = dbpedia_dir / '_VALID'

    if not flag.exists():
        from torchtext.datasets import DBpedia

        dbpedia_dir.mkdir(exist_ok=True, parents=True)

        train = list(DBpedia(root=root, split='train'))
        test = list(DBpedia(root=root, split='test'))
        random.shuffle(train)
        random.shuffle(test)

        test = test[:1000]
        train = train[:10000]

        ten_per = int(0.1 * len(train))
        valid, train = train[:ten_per], train[ten_per:]

        fargs = dict(mode='w', encoding='utf8', errors='ignore')
        for name, data in [('train', train), ('valid', valid), ('test', test)]:
            log.info(f"Writing {name}")
            text_f = dbpedia_dir / f'{name}.text'
            label_f = dbpedia_dir / f'{name}.label'
            with text_f.open(**fargs) as text, label_f.open(**fargs) as label:
                for l, t in data:
                    text.write(t.replace("\n", " ").replace("\t", " ").strip() + '\n')
                    label.write(f'{l}\n')
        flag.touch()


def test_tfmcls_model():
    try:
        setup_dataset()
    except Exception as e:
        log.error(e)
        return

    tmp_dir = tempfile.mkdtemp()
    try:
        config = load_conf('tests/experiments/transformer-classifier.conf.yml')
        model_type = config['model_type']
        # tmp_dir = Path('tmp.dbpedia-exp')
        exp = MODELS[model_type].Experiment(tmp_dir, config=config, read_only=False)

        exp.config['trainer'].update(dict(steps=1000, check_point=250))
        # exp.config['prep']['num_samples'] = 0
        Pipeline(exp).run(run_tests=False)
        sanity_check_experiment(exp, samples=False, shared_vocab=False)
        print(f"Cleaning up {tmp_dir}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)