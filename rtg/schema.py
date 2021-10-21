#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 8/4/21

from rtg import log
from rtg.registry import CRITERION, OPTIMIZER


def config_checks(config):
    # these are required for training models
    if 'optim' in config or OPTIMIZER not in config or CRITERION not in config:
        help_url = 'https://github.com/isi-nlp/rtg-in/issues/260'
        log.warning("Kindly migrate the 'optim' config block to new and improved schema")
        log.info(f"For migration info visit {help_url}")
        raise ValueError(f'Config migration to new version is required; see {help_url};\n '
                         f'You have {dict(config).keys()}; expected {OPTIMIZER}, {CRITERION} but not optim')
