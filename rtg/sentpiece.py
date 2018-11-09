#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 11/7/18

import argparse
import sys
from sentencepiece import SentencePieceProcessor
from typing import List


PAD_TOK = '<pad>', 0
UNK_TOK = '<unk>', 1
BOS_TOK = '<s>', 2
EOS_TOK = '</s>', 3


class Field(SentencePieceProcessor):
    """A wrapper class for sentence piece trainer and processor"""

    def __init__(self, path):
        super(Field, self).__init__()
        assert self.load(path)

    def tokenize(self, text: str, add_bos=True, add_eos=True) -> List[bytes]:
        text = text.strip()
        if not text:
            return []
        pieces: List[bytes] = self.encode_as_pieces(text.encode())
        if add_bos and pieces[0] != BOS_TOK[0]:
            pieces.insert(0, BOS_TOK[0].encode())
        if add_eos and pieces[-1] != EOS_TOK[0]:
            pieces.append(EOS_TOK[0].encode())
        return pieces

    def encode_as_ids(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        ids = super(Field, self).encode_as_ids(text)
        if add_bos and ids[0] != BOS_TOK[1]:
            ids.insert(0, BOS_TOK[1])
        if add_eos and ids[-1] != EOS_TOK[1]:
            ids.append(EOS_TOK[1])
        return ids

    @staticmethod
    def detokenize(tokens: List[str], remove_bos=True, remove_eos=True) -> str:
        if remove_bos and tokens[0] == BOS_TOK[0]:
            tokens = tokens[1:]
        if remove_eos and tokens[-1] == EOS_TOK[0]:
            tokens = tokens[:-1]
        text = ''.join(tokens).replace('▁', ' ').strip()
        return text

    def decode_ids(self, ids: List[int], trunc_eos=False) -> str:
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        if ids[0] == BOS_TOK[1]:
            ids = ids[1:]
        if trunc_eos:
            try:
                ids = ids[:ids.index(EOS_TOK[1])]
            except ValueError:
                pass
        return super(Field, self).decode_ids(ids)


def main(model, inp, out, ids, detokenize):
    spm = Field(model)
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
                    pieces = [str(x, encoding='utf-8') for x in pieces]
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

