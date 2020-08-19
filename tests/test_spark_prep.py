#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 8/2/20


import pytest
from rtg.pipeline import Pipeline, Experiment
import tempfile
from rtg.exp import load_conf
import shutil
from . import sanity_check_experiment


def test_spark_prep():
    tmp_dir = tempfile.mkdtemp()
    try:
        print(f"Testing dataprep on pyspark: {tmp_dir}")
        config = load_conf('experiments/spark-bigdataprep.yml')
        exp = Experiment(tmp_dir, config=config, read_only=False)
        exp.config['trainer'].update(dict(steps=50, check_point=25, batch_size=400))
        Pipeline(exp).run()
        assert exp._prepared_flag.exists()
        assert exp._trained_flag.exists()
        assert exp.train_file.exists() or exp.train_db.exists()
        sanity_check_experiment(exp)
    finally:
        print(f"Cleaning up {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


