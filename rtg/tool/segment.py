#!/usr/bin/env python
#
# An interactive CLI interface to sentence piece processor
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/19/19

import argparse
import sys
from rtg import TranslationExperiment as Experiment, log


def run_all(exp: Experiment, inp, side: str='shared', is_ids: bool = False, is_merge: bool = False):
    """

    :param exp: Experiment
    :param inp: input to read from
    :param side: which side of vocabulary
    :param is_ids: encode or decode ids
    :param is_merge: decode (i.e. merge) not encode (i.e. split)
    :return:
    """
    vocab = {'src': exp.src_vocab, 'tgt': exp.tgt_vocab, 'shared': exp.src_vocab}[side]
    func = {
        # (is_merge, is_ids) : func
        (False, False): vocab.encode_as_pieces,
        (False, True): vocab.encode_as_ids,
        (True, False): vocab.detokenize,
        (True, True): vocab.decode_ids
    }[is_merge, is_ids]
    log.info(f"Reading from {inp.name}")
    for line in inp:

        # prep
        rec = line.strip()
        if is_merge:
            rec = line.split()
            if is_ids:
                rec = [int(x) for x in rec]

        rec = func(rec)

        # post prep
        line = rec
        if not is_merge:
            if is_ids:
                line = map(str, rec)
            line = ' '.join(line)
        yield line


def write_all(lines, out):
    """
    Writes all out puts to
    :param lines: stream of records
    :param out:output for writing
    :return:
    """
    log.info(f"writing to {out.name}")
    for line in lines:
        out.write(line + '\n')


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp', help='Path to experiment which has vocabulary files under <exp>/data/')
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    p.add_argument('-s', '--side', default='shared', choices={'src', 'tgt', 'shared'},
                   help='vocabulary side;')
    p.add_argument('--ids', dest='is_ids', action='store_true',
                   help='Word Ids instead of pieces')
    p.add_argument('--merge', dest='is_merge', action='store_true',
                   help='Merge or detokenize or decode or undo splits')
    args = vars(p.parse_args())
    args['exp'] = Experiment(args.pop('exp'), read_only=True)
    out = args.pop('out')
    recs = run_all(**args)
    write_all(recs, out)


if __name__ == '__main__':
    main()
