#!/usr/bin/env python
#
#
# Author: Thamme Gowda
# Created: 12/23/21

from rtg.distrib import launch, log
import tempfile
import shutil


def test_distrib_train():
    sample_exp = 'experiments/sample-exp'
    with tempfile.TemporaryDirectory() as tmp_dir:
        log.info(f"Copy: {sample_exp} --> {tmp_dir}")
        shutil.copytree(src=sample_exp, dst=tmp_dir, dirs_exist_ok=True)

        # 1 node, 3 processes, no GPU, mggodule rtg.pipeline
        args = f'-N 1 -P 3 -G 0 -m rtg.pipeline {tmp_dir}'.split()
        p_args = launch.parse_args(args)
        launch.main(p_args)

