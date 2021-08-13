#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 1/24/19
from rtg.exp import TranslationExperiment as Experiment
from dataclasses import dataclass
from pathlib import Path
from rtg.module.decoder import Decoder
from rtg import log, device, yaml
from rtg.utils import IO
import datetime

import os
import torch
import time
import argparse

@dataclass
class ExperimentExporter:
    exp: Experiment

    def export(self, target: Path, name: str=None, ensemble: int = 1, copy_config=True,
               copy_vocab=True):
        to_exp = Experiment(target.resolve(), config=self.exp.config)

        if copy_config:
            log.info("Copying config")
            to_exp.persist_state()

        if copy_vocab:
            log.info("Copying vocabulary")
            self.exp.copy_vocabs(to_exp)
        assert ensemble > 0
        assert name
        assert len(name.split()) == 1
        log.info("Going to average models and then copy")
        model_paths = self.exp.list_models()[:ensemble]
        log.info(f'Model paths: {model_paths}')
        chkpt_state = torch.load(model_paths[0], map_location=device)
        if ensemble > 1:
            log.info("Averaging them ...")
            avg_state = self.exp.average_states(model_paths)
            chkpt_state = dict(model_state=avg_state,
                               model_type=chkpt_state['model_type'],
                               model_args=chkpt_state['model_args'])
        log.info("Instantiating it ...")
        model = self.exp.load_model_with_state(checkpt_state=chkpt_state)
        log.info(f"Exporting to {target}")
        to_exp = Experiment(target, config=self.exp.config)
        to_exp.persist_state()
        
        IO.copy_file(self.exp.model_dir / 'scores.tsv', to_exp.model_dir / 'scores.tsv')
        if (self.exp.work_dir / 'rtg.zip').exists():
            IO.copy_file(self.exp.work_dir / 'rtg.zip', to_exp.work_dir / 'rtg.zip')
        
        src_chkpt = chkpt_state
        log.warning("step number, training loss and validation loss are not recalculated.")
        step_num, train_loss, val_loss = [src_chkpt.get(n, -1)
                                          for n in ['step', 'train_loss', 'val_loss']]
        copy_fields = ['optim_state', 'step', 'train_loss', 'valid_loss', 'time',
                       'rtg_version', 'model_type', 'model_args']
        state = dict((c, src_chkpt[c]) for c in copy_fields if c in src_chkpt)
        state['model_state'] = model.state_dict()
        state['averaged_time'] = time.time()
        state['model_paths'] = model_paths
        state['num_checkpts'] = len(model_paths)
        prefix = f'model_{name}_avg{len(model_paths)}'
        to_exp.store_model(step_num, state, train_score=train_loss, val_score=val_loss, keep=10,
                           prefix=prefix)
        chkpts = [mp.name for mp in model_paths]
        status = {
            'parent': str(self.exp.work_dir),
            'ensemble': ensemble,
            'checkpts': chkpts,
            'when': datetime.datetime.now().isoformat(),
            'who': os.environ.get('USER', '<unknown>'),
        }
        yaml.dump(status, stream=to_exp.work_dir / '_EXPORTED')

        if self.exp._trained_flag.exists():
            IO.copy_file(self.exp._trained_flag, to_exp._trained_flag)


def add_boolean(parser, name, help, dest=None, default=True):
    group = parser.add_mutually_exclusive_group()
    dest = dest if dest else name.lower().replace('-', '_')
    group.add_argument(f'--{name}', dest=dest, action='store_true', default=default, help=help)
    group.add_argument(f'--no-{name}', dest=dest, action='store_false', default=not default,
                       help=f"See --{name}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source', metavar='DIR', type=Path, help='Path to experiment (source)')
    p.add_argument('target', metavar='DIR', type=Path,
                   help='Path to destination where the export should be')
    p.add_argument('-en', '--ensemble', metavar='N', type=int, default=5,
                   help='Maximum number of checkpoints to average and export. -en=1 to get just 1.')
    p.add_argument('-nm', '--name', required=True,
                   help='Name for the exported model.'
                        'Value should be a single word. This will be useful if you are going to '
                        'place multiple exports in a same dir for system combination')
    add_boolean(p, 'config', dest='copy_config', help='Copy config')
    add_boolean(p, 'vocab', dest='copy_vocab',
                help='Copy vocabulary files (such as sentence piece models)')
    args = vars(p.parse_args())
    return args


def main():
    args = parse_args()
    tool = ExperimentExporter(Experiment(args.pop('source')))
    tool.export(**args)


if __name__ == '__main__':
    main()
