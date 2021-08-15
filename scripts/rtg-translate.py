#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 8/10/21


import logging as log
import requests
from typing import List, Iterator, Union
from tqdm import tqdm
import json


log.basicConfig(level=log.INFO)
DEF_API = "https://localhost:6060/translate"
DEF_BATCHSIZE = 10

class RTGClient:

    def __init__(self, api_url: str):
        log.info(f"Creating RTG API Client for {api_url}")
        self.api_url = api_url

    def translate(self, sents: List[str]):
        assert isinstance(sents, list)
        assert len(sents) > 0
        assert isinstance(sents[0], str)
        sents = [s.strip() or '.' for s in sents]    # insert dot for empty

        data = {'source': sents}
        resp = requests.post(self.api_url, json=data)
        if resp.status_code != 200:
            log.warning(f"Oops! something went wrong. Check logs. See if {self.api_url} is valid")
        result = resp.json()
        result = result['translation']
        assert len(result) == len(sents)
        return result

    def translate_all(self, sents: Union[List[str], Iterator[str]], batch_size: int,
                      tsv_mode=False):
        buffer = []
        ids = []
        total = len(sents) if hasattr(sents, '__len__') else None
        log.info(f"Translating: batch_size {batch_size}; total={total or 'unknown'}")
        for sent in tqdm(sents, total=total):
            if tsv_mode:
                id, sent = sent.split('\t')
                ids.append(id)
            buffer.append(sent)
            if len(buffer) >= batch_size:
                result = self.translate(buffer)
                if tsv_mode:
                    assert len(ids) == len(buffer)
                    result = [f'{id}\t{txt}' for id, txt in zip(ids, result)]
                    ids.clear()
                yield from result
                buffer.clear()

        if buffer:
            result = self.translate(buffer)
            if tsv_mode:
                assert len(ids) == len(buffer)
                result = [f'{id}\t{txt}' for id, txt in zip(ids, result)]
                ids.clear()
            yield from result


def main(**args):
    args = args or vars(parse_args())
    client = RTGClient(api_url=args['api'])
    sents = args['inp']

    result = client.translate_all(sents=sents, batch_size=args['batch_size'],
                                  tsv_mode=args.get('tsv'))
    out = args['out']
    count = 1
    for sent in result:
        out.write(f'{sent}\n')
        count += 1
    log.info(f"Wrote {count} lines to {out}")

def parse_args():
    import argparse
    import sys
    import io
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-a', '--api', default=DEF_API, help='API URL')
    p.add_argument('-b', '--batch-size', default=DEF_BATCHSIZE, help='Batch size')
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=stdin,
                   help='Input file path')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=stdout,
                   help='Output file path')
    p.add_argument('-tsv', '--tsv', action='store_true', help='Input is TSV of <id>\\t<text>')

    return p.parse_args()


if __name__ == '__main__':
    main()
