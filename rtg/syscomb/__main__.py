#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 1/3/19

import argparse
from pathlib import Path
from rtg.syscomb import SysCombTrainer


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('experiment', type=Path, help='Path to experiment directory')
    p.add_argument('models', nargs='+', help="Path to models", type=Path)
    p.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size')
    p.add_argument('-s', '--steps', type=int, default=2000, help='Training steps')
    args = vars(p.parse_args())
    assert len(args['models']) > 1, 'At least two models should be given for system combination'
    trainer = SysCombTrainer(models=args.pop('models'), exp=args.pop('experiment'))
    trainer.train(**args)


if __name__ == '__main__':
    main()
