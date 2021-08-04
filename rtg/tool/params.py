#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/30/21

import logging as log
from pathlib import Path
from rtg.exp import TranslationExperiment
from rtg.module.decoder import  instantiate_model
import torch

log.basicConfig(level=log.INFO)


def main(args=None):
    args = args or parse_args()
    exp = TranslationExperiment(args['exp'], read_only=True)
    model_path, step = exp.get_last_saved_model()
    assert model_path
    assert model_path.exists()
    state = torch.load(model_path, map_location='cpu')
    model = instantiate_model(state, exp=exp)
    print(model)

    total = 0
    table = []
    justify = 0
    for name, param in model.named_parameters():
        n = param.numel()
        if not param.requires_grad:
            print(f"Skip {name}")
            continue
        justify = max(justify, len(name))
        total += n
        table.append((name, n, str(tuple(param.size()))))

    justify2 = len(f'{total:,}') + 1
    print("======")
    print("\n".join(f'{name: <{justify}}\t{val: {justify2},}\t{shape}' for name, val, shape in table))
    print(f"\n\n{'Total': <{justify}}\t{total: {justify2},}\t")

def parse_args():
    import argparse

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp',  type=Path, help='Path to Experiment')
    return vars(p.parse_args())


if __name__ == '__main__':
    main()
