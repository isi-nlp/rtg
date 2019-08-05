#!/usr/bin/env python
# Unicode eXpander and De-eXpander (uxdx)
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-06-30

import argparse
import sys
from typing import List, Set, Union, Dict, Optional, Iterator


def codes_to_str(codes: List[int]) -> str:
    return ''.join(map(chr, codes))


class Devanagari:

    def __init__(self):
        self.consonants = set(range(ord('क'), ord('ह') + 1)) | set(range(ord('क़'), ord('य़') + 1))
        self.impl_vowel_form = ord('⋮')
        self.virams = {ord('्')}

    def decode(self, line: str) -> str:
        codes = list(map(ord, line))
        dec = (c for c in codes if c != self.impl_vowel_form)
        return codes_to_str(dec)

    def encode(self, line: str) -> str:
        codes = list(map(ord, line))
        enc = []
        for i, c in enumerate(codes):
            # if codes[i] in self.virams: continue # skip  virama
            enc.append(c)
            if c in self.consonants and i + 1 < len(codes) \
                    and codes[i + 1] in self.consonants:
                enc.append(self.impl_vowel_form)
        return codes_to_str(enc)


def main(inp, out, decode=False):
    lang = Devanagari()
    lines = (line.rstrip('\n') for line in inp)
    lines = map(lang.decode if decode else lang.encode, lines)
    for line in lines:
        out.write(line)
        out.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r', encoding='utf8', errors='ignore'),
                   default=sys.stdin, help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w', encoding='utf8', errors='ignore'),
                   default=sys.stdout,
                   help='Output file path')
    p.add_argument('-d', '--decode', action='store_true',
                   help="Restore the standard unicode stream")
    # p.add_argument('-l', '--lang', choices=['hin', 'devanagari'], help="Language identifier", required=True)
    args = vars(p.parse_args())
    main(**args)
