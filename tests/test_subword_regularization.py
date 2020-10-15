#!/usr/bin/env python
#
# Author: Lukas Ferrer [lferrer (at) isi (dot) edu]
# Created: 7/13/2020
import pytest
from rtg.pipeline import Pipeline, Experiment
import tempfile
from rtg.exp import load_conf
import shutil


def test_finetune_pipeline_transformer():
    codec_lib = 'nlcodec'
    tmp_dir = tempfile.mkdtemp()
    print(f"Testing finetune transformer: {tmp_dir}")
    config = load_conf('experiments/sample-exp/conf.yml')
    prep = config['prep']
    prep.update(dict(codec_lib=codec_lib, char_coverage=0.9995,
                     finetune_src=prep['train_src'],
                     finetune_tgt=prep['train_tgt']))
    exp = Experiment(tmp_dir, config=config, read_only=False)
    exp.config['trainer'].update(dict(
        steps=50, check_point=25, finetune_steps=100, batch_size=400, split_ratio=0.1, dynamic_epoch=True
    ))
    Pipeline(exp).run()
    assert exp.train_file.exists() or exp.train_db.exists()
    assert exp.finetune_file.exists()
    # TODO: add more assertions

    print(f"Cleaning up {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
