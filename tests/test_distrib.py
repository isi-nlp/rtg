#!/usr/bin/env python
#
#
# Author: Thamme Gowda
# Created: 12/23/21

import tempfile
import shutil

from rtg import log, cpu_count
from rtg.cli import launch
import os


def test_distrib_train():
    sample_exp = 'experiments/sample-exp'
    with tempfile.TemporaryDirectory() as tmp_dir:
        log.info(f"Copy: {sample_exp} --> {tmp_dir}")
        shutil.copytree(src=sample_exp, dst=tmp_dir, dirs_exist_ok=True)
        os.environ['OMP_NUM_THREADS'] = str(cpu_count)
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        # 1 node, 3 processes, no GPU, module rtg.pipeline
        args = f'-N 1 -P 2 -G 0 -m rtg.cli.pipeline {tmp_dir}'.split()
        p_args = launch.parse_args(args)
        launch.main(p_args)
