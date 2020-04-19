#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-07-26

import argparse
import sys
import logging as log
from pathlib import Path
from typing import TextIO
from rtg.data.dataset import SqliteFile

log.basicConfig(level=log.INFO)


def main(db_path: Path, field: str, out: TextIO):
    assert db_path.exists(), f'{db_path} should exist'
    db = SqliteFile(db_path, max_src_len=10_000, max_tgt_len=10_000)
    for rec in db.read_all():
        val = rec[field]
        if isinstance(val, list):
            val = " ".join(map(str, val))
        else:
            val = str(val)
        out.write(f"{val}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('db_path', type=Path, help='Input file path to db')
    p.add_argument('field', choices={"x", "y", "x_len", "y_len"}, help='field name to extract')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    args = vars(p.parse_args())
    main(**args)
