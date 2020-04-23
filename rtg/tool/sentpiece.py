#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/7/18

import argparse
import sys
from rtg.data.codec import SPField

ADD_BOS = False
ADD_EOS = False


def main(model, inp, out, ids, detokenize):
    spm = SPField(model)
    count = 0
    try:
        for line in inp:
            count += 1
            if detokenize:
                pieces = line.strip().split()
                if ids:
                    piece_ids = [int(x) for x in pieces]
                    out_line = spm.decode_ids(piece_ids) + '\n'
                else:
                    out_line = spm.detokenize(pieces) + '\n'
            else:
                if ids:
                    piece_ids = spm.encode_as_ids(line)
                    pieces = [str(x) for x in piece_ids]
                else:
                    pieces = spm.tokenize(line)
                    pieces = [x if type(x) is str else str(x, encoding='utf-8') for x in pieces]
                out_line = " ".join(pieces) + "\n"
            out.write(out_line)
        msg = ('Detokenized' if detokenize else 'Tokenized') + f" lines={count}"
        print(msg, file=sys.stderr)
    except Exception as e:
        print(f"Error at line number={count}", file=sys.stderr)
        raise e


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-m', '--model', required=True, help='Sentence piece model path')
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    p.add_argument('-id', '--ids', action='store_true',
                   help='Ids (integers) instead of text pieces (strings)')
    p.add_argument('-d', '--detokenize', action='store_true', help='Detokenize or undo tokenize')

    args = vars(p.parse_args())
    main(**args)

