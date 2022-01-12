#!/usr/bin/env python
#
#
# Author: Thamme Gowda
# Created: 1/11/22

import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
from pathlib import Path
from typing import TextIO, Tuple

import torch
from torch import Tensor

from rtg import my_tensor as tensor
from rtg.data.dataset import Batch as TrainerBatch
from rtg.exp import TranslationExperiment
from rtg.exp import load_conf
from rtg.module.decoder import Decoder
from rtg.registry import registry, MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Force decode and attention visualization",
                                     formatter_class=ArgFormatter)
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument("-s", "--src", type=argparse.FileType('r', encoding='utf-8'), default=sys.stdin,
                        help="Source file having <seg1><delim><seg2>. Default: STDIN")
    parser.add_argument("-r", "--ref", type=argparse.FileType('r', encoding='utf-8'), required=True,
                        help="Reference file having <ref1><delim><ref2>")
    parser.add_argument("-d", "--delim", default='\t', help="Delimiter: default:\\t")

    parser.add_argument("-of", '--output', default=sys.stdout, nargs='*',
                        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        help='Output File path. default is STDOUT')
    parser.add_argument("-b", '--batch-size', type=int, help='batch size; number of sentences')
    parser.add_argument("-msl", '--max-src-len', type=int,
                        help='max source len; longer seqs will be truncated')
    parser.add_argument("-nb", '--no-buffer', action='store_true',
                        help='Processes one line per batch followed by flush output')
    args = vars(parser.parse_args())
    return args


def make_batches(recs, batch_size=10):
    res = [[]]
    for rec in recs:
        if len(res[-1]) > batch_size:
            res.append([])
        res[-1].append(rec)
    # separate columns
    n_cols = len(recs[0])
    res2 = []
    for b in res:
        res2.append([])
        for i in range(n_cols):
            res2[-1].append([x[i] for x in b])
    return res2


def force_decode(decoder, src, ref):
    model = decoder.model
    assert model.cache_attn
    src_seq = decoder.inp_vocab.encode_as_ids(src, add_eos=True, add_bos=False)
    ref_seq = decoder.out_vocab.encode_as_ids(ref, add_eos=True, add_bos=True)
    src_seqs = tensor(src_seq, dtype=torch.long).view(1, -1)
    tgt_seqs = tensor(ref_seq, dtype=torch.long).view(1, -1)

    tgt_in_seqs = tgt_seqs[:, :-1]  # skip EOS
    x_mask = (src_seqs != decoder.inp_vocab.pad_idx).unsqueeze(1)
    y_mask = TrainerBatch.make_autogres_mask_(tgt_in_seqs, decoder.out_vocab.pad_idx)

    out_feats = model(src_seqs, tgt_in_seqs, x_mask, y_mask)
    # self.model.generator(out_feats)
    x_probs = model.generator(out_feats, score='log_probs')  # B=1 x T x V
    out_ids = tgt_seqs[0, 1:]  # Skip BOS  # [T]
    x_probs = x_probs.squeeze(0)  # T x V
    force_score = x_probs.gather(1, out_ids.view(-1, 1))
    score = force_score.sum().item()
    attns = [model.encoder.self_attn, model.decoder.self_attn, model.decoder.src_attn]
    attns = [a[0] for a in attns]  # it was one sentence only
    return score, attns, (src_seq, ref_seq)


def compute_attn_bleed(attn: Tensor, cross_over: Tuple[int, int], epsilon=1e-6):
    n_tgt, n_src = attn.shape
    src_x, tgt_x = cross_over
    assert 0 < tgt_x < n_tgt
    assert 0 < src_x < n_src
    good_mass, bad_mass = 0, 0
    for ti, row in enumerate(attn):
        assert abs(row.sum() - 1) < epsilon  # row sums to 1
        part1, part2 = row[:src_x], row[src_x:]  # upto src_x is first sentence; after is second sentence
        if ti > tgt_x:
            part1, part2 = part2, part1
        good_mass += part1.sum()
        bad_mass += part2.sum()
    total_mass = n_tgt  # and each has mass of 1.0
    assert abs(total_mass - good_mass - bad_mass) - epsilon
    return bad_mass / total_mass  # and that is bleed ratio, finally


def avg_attn_bleed(attn, cross_over: Tuple[int, int]):
    n_layers, n_heads, n_ref, n_src = attn.shape
    res = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        for head in range(n_heads):
            res[layer, head] = compute_attn_bleed(attn[layer, head], cross_over)
    return res.mean(dim=1).mean().item()  # mean over heads, then mean over layers


def decode_recs(recs, decoder: Decoder, output: TextIO):
    rates = []
    for src, ref, src_idx, ref_idx in recs:
        score, attns, (src_seq, ref_seq) = force_decode(decoder=decoder, src=src, ref=ref)
        enc_attn, dec_attn, xattn = attns
        rate = avg_attn_bleed(xattn, (src_idx, ref_idx))
        output.write(f'{rate:.4f}\n')
        rates.append(rate)
    # mean over the corpus
    return sum(rates) / len(rates)


def compute_bleed(exp, **cli_args):
    dec_args = exp.config.get('decoder') or exp.config['tester'].get('decoder', {})
    src_file: TextIO = cli_args.pop('src')
    ref_file: TextIO = cli_args.pop('ref')
    delim: str = cli_args.pop('delim')
    joiner: str = cli_args.pop('joiner', ' ')

    output: TextIO = cli_args.pop('output')
    decoder = Decoder.new(exp, ensemble=dec_args.pop('ensemble', 1))
    if not decoder.model.cache_attn:
        decoder.model.cache_attn = True

    # we need to find length where  merging of two segments happen
    def _prep_input():
        for src, ref in zip(src_file, ref_file):
            assert isinstance(src, str) and delim in src
            assert isinstance(ref, str) and delim in ref
            srcs = src.split(delim)
            refs = ref.split(delim)
            assert len(srcs) == len(refs) == 2
            src1_ids = decoder.inp_vocab.encode_as_ids(srcs[0], add_bos=False, add_eos=False)
            ref1_ids = decoder.out_vocab.encode_as_ids(refs[0], add_bos=True, add_eos=False)
            rec = (joiner.join(srcs), joiner.join(refs), len(src1_ids), len(ref1_ids))
            yield rec

    bleed_rate = decode_recs(_prep_input(), decoder, output)
    return bleed_rate


def main(**args):
    # No grads required for decode
    torch.set_grad_enabled(False)
    cli_args = args or parse_args()
    exp_dir = Path(cli_args.pop('exp_dir'))
    conf = load_conf(exp_dir / 'conf.yml')
    assert conf.get('model_type')
    exp_factory = TranslationExperiment
    if conf['model_type'] in registry[MODEL]:
        exp_factory = registry[MODEL][conf['model_type']].Experiment
    exp = exp_factory(exp_dir, config=conf, read_only=True)
    assert isinstance(exp, TranslationExperiment)
    rate = compute_bleed(exp, **cli_args)
    print(rate)


if __name__ == '__main__':
    main()
