#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/18/20
import pytest
from rtg.pipeline import Pipeline, Experiment
import tempfile
from rtg.exp import load_conf
import torch
import shutil


def test_prepared_pipeline():
    exp = Experiment('experiments/sample-exp', read_only=True)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    pipe = Pipeline(exp)
    pipe.run(run_tests=False)


def sanity_check_experiment(exp):
    """
    Bunch of sanity checks on experiment dir
    :param exp:
    :return:
    """
    assert exp._config_file.exists()
    assert exp.data_dir.exists()
    assert exp.train_db.exists()
    assert exp.train_db.stat().st_size > 0
    assert exp.valid_file.exists()
    assert exp.samples_file.exists()
    assert exp._shared_field_file.exists()
    assert exp._prepared_flag.exists()
    assert exp._trained_flag.exists()
    assert len(list(exp.model_dir.glob('*.pkl'))) > 0
    assert len((exp.model_dir / 'scores.tsv').read_text().splitlines()) > 0


def test_pipeline_transformer():
    for codec_lib in ['sentpiece', 'nlcodec']:
        tmp_dir = tempfile.mkdtemp()
        config = load_conf('experiments/transformer.test.yml')
        print(f"Testing {codec_lib} --> {tmp_dir}")
        config['prep'].update({'codec_lib': codec_lib, 'char_coverage': 0.9995})
        exp = Experiment(tmp_dir, config=config, read_only=False)
        exp.config['trainer'].update(dict(steps=50, check_point=25))
        Pipeline(exp).run(run_tests=False)
        sanity_check_experiment(exp)
        print(f"Cleaning up {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This is too slow on CPU")
def test_robertamt_full_init():
    tmp_dir = tempfile.mkdtemp()
    config = load_conf('experiments/pretrained/robertamt-xlmr.yml')
    model_id = config['model_args']['model_id']
    print(f"Testing {model_id} --> {tmp_dir}")
    assert 'pretrainmatch' == config['prep'].get('codec_lib')
    exp = Experiment(tmp_dir, config=config, read_only=False)
    exp.config['trainer'].update(dict(steps=4, check_point=1))
    Pipeline(exp).run(run_tests=False)
    sanity_check_experiment(exp)
    print(f"Cleaning up {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This is too slow on CPU")
def test_robertamt_2layer_init():
    tmp_dir = tempfile.mkdtemp()
    config = load_conf('experiments/pretrained/robertamt-xlmr-2layer.yml')
    model_id = config['model_args']['model_id']
    print(f"Testing {model_id} --> {tmp_dir}")
    assert 'pretrainmatch' == config['prep'].get('codec_lib')
    exp = Experiment(tmp_dir, config=config, read_only=False)
    exp.config['trainer'].update(dict(steps=4, check_point=1))
    Pipeline(exp).run(run_tests=False)
    sanity_check_experiment(exp)
    print(f"Cleaning up {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_parent_child_pipeline():
    parent_dir = tempfile.mkdtemp()
    # parent_dir = 'tmp-xyz-parent'

    print(f"Making parent at {parent_dir}")
    exp = Experiment(parent_dir, config='experiments/transformer.test.yml', read_only=False)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    Pipeline(exp).run(run_tests=False)
    sanity_check_experiment(exp)
    assert not exp.parent_model_state.exists()

    child_config = load_conf('experiments/transformer.test.yml')
    child_config.update({
        'parent': {
            'experiment': str(parent_dir),
            'vocab': {
                'shared': 'shared'
            },
            'model': {
                'ensemble': 2
            }
        }
    })

    child_dir = tempfile.mkdtemp()
    # child_dir = 'tmp-xyz-child'
    print(f"Making child at {child_dir}")
    exp = Experiment(child_dir, config=child_config, read_only=False)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    Pipeline(exp).run(run_tests=False)
    sanity_check_experiment(exp)
    assert exp.parent_model_state.exists()

    for dir in [parent_dir, child_dir]:
        print(f"Cleaning up {dir}")
        shutil.rmtree(dir, ignore_errors=True)


def test_freeze_pipeline():
    exp = Experiment('experiments/sample-exp', read_only=True)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    # enable these
    trainable = {'include': ['src_embed', 'tgt_embed', 'generator', 'encoder:0', 'decoder:0,1']}
    exp.config['optim']['trainable'] = trainable
    pipe = Pipeline(exp)
    pipe.run(run_tests=False)