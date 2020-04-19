#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/18/20
import pytest


def test_prepared_pipeline():
    from rtg.pipeline import Pipeline, Experiment
    exp = Experiment('experiments/sample-exp', read_only=True)
    exp.config['trainer'].update(dict(steps=200, check_point=200))
    pipe = Pipeline(exp)
    pipe.run(run_tests=False)

def test_pipeline_transformer():
    from rtg.pipeline import Pipeline, Experiment
    import tempfile
    from rtg.exp import load_conf
    import shutil
    for codec_lib in ['sentpiece', 'nlcodec']:
        tmp_dir = tempfile.mkdtemp()
        config = load_conf('experiments/transformer.test.yml')
        print(f"Testing {codec_lib} --> {tmp_dir}")
        config['prep'].update({'codec_lib':  codec_lib, 'char_coverage': 0.9995})
        exp = Experiment(tmp_dir, config=config, read_only=False)
        exp.config['trainer'].update(dict(steps=200, check_point=200))
        Pipeline(exp).run(run_tests=False)
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
        print(f"Cleaning up {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


