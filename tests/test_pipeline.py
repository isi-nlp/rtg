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
from io import StringIO
from . import sanity_check_experiment


def test_prepared_pipeline():
    exp = Experiment('experiments/sample-exp', read_only=True)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    pipe = Pipeline(exp)
    pipe.run(run_tests=False)


def test_pipeline_transformer():

    def _run_decode(exp_dir, sentences):
        assert isinstance(sentences, list)
        from rtg.decode import main as decode_cli
        buffer = StringIO()
        decode_cli(exp_dir=exp_dir, input=[sentences], output=[buffer], skip_check=True, max_src_len=200)
        lines = buffer.getvalue().splitlines()
        buffer.close()
        return lines

    for codec_lib in ['sentpiece', 'nlcodec']:
        tmp_dir = tempfile.mkdtemp()
        config = load_conf('experiments/transformer.test.yml')
        print(f"Testing {codec_lib} --> {tmp_dir}")
        config['prep'].update({'codec_lib': codec_lib, 'char_coverage': 0.9995})
        exp = Experiment(tmp_dir, config=config, read_only=False)
        exp.config['trainer'].update(dict(steps=50, check_point=25))
        exp.config['prep']['num_samples'] = 0
        Pipeline(exp).run(run_tests=False)
        sanity_check_experiment(exp)
        print(f"Cleaning up {tmp_dir}")
        src_sents = ["hello there", "this is a test"]
        output = _run_decode(exp_dir=tmp_dir, sentences=src_sents)
        assert len(src_sents) == len(output)
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


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()   # required for parallel nlcodec
    #test_pipeline_transformer()
    pass