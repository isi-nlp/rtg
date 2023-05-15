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

from rtg import device
from rtg import my_tensor as tensor
from rtg.data.dataset import Batch as TrainerBatch
from rtg.exp import TranslationExperiment, load_conf
from rtg.module.decoder import Decoder
from rtg.registry import MODEL, registry

MAX_LEN = 640
BATCH_SIZE = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Force decode and attention visualization", formatter_class=ArgFormatter
    )
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument(
        "-s",
        "--src",
        type=argparse.FileType('r', encoding='utf-8'),
        default=sys.stdin,
        help="Source file having <seg1><delim><seg2>. Default: STDIN",
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=argparse.FileType('r', encoding='utf-8'),
        required=True,
        help="Reference file having <ref1><delim><ref2>",
    )
    parser.add_argument("-d", "--delim", default='\t', help="Delimiter: default:\\t")

    parser.add_argument(
        "-of",
        '--output',
        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
        default=sys.stdout,
        help='Output File path. default is STDOUT',
    )
    parser.add_argument(
        "-b", '--batch-size', type=int, default=BATCH_SIZE, help='batch size; number of sentences'
    )
    args = vars(parser.parse_args())
    return args


def make_batches(recs, batch_size=BATCH_SIZE):
    def separate_cols(b):
        n_cols = len(b[0])
        return [[x[a] for x in b] for a in range(n_cols)]

    buffer = []
    for rec in recs:
        if len(buffer) > batch_size:
            yield separate_cols(buffer)
            buffer = []
        buffer.append(rec)
    if buffer:
        yield separate_cols(buffer)


def get_attns(decoder, srcs, refs, max_len=MAX_LEN):
    model = decoder.model
    assert model.cache_attn
    n_seqs = len(srcs)
    src_seqs_list = [
        decoder.inp_vocab.encode_as_ids(src, add_eos=True, add_bos=False)[:max_len] for src in srcs
    ]
    tgt_seqs_list = [
        decoder.out_vocab.encode_as_ids(ref, add_eos=True, add_bos=True)[:max_len] for ref in refs
    ]

    max_src_len = max(len(s) for s in src_seqs_list)
    max_tgt_len = max(len(s) for s in tgt_seqs_list)
    src_seqs = torch.full((n_seqs, max_src_len), device=device, fill_value=decoder.inp_vocab.pad_idx)
    tgt_seqs = torch.full((n_seqs, max_tgt_len), device=device, fill_value=decoder.out_vocab.pad_idx)
    for i in range(n_seqs):
        src = src_seqs_list[i]
        tgt = tgt_seqs_list[i]
        src_seqs[i, : len(src)] = tensor(src)
        tgt_seqs[i, : len(tgt)] = tensor(tgt)

    tgt_in_seqs = tgt_seqs[:, :-1]  # skip EOS
    x_mask = (src_seqs != decoder.inp_vocab.pad_idx).unsqueeze(1)
    y_mask = TrainerBatch.make_autogres_mask_(tgt_in_seqs, decoder.out_vocab.pad_idx)
    out_feats = model(src_seqs, tgt_in_seqs, x_mask, y_mask)
    attns = [model.encoder.self_attn, model.decoder.self_attn, model.decoder.src_attn]
    return attns, (src_seqs_list, tgt_seqs_list)


"""
def compute_attn_bleed(attn: Tensor, cross_over: Tuple[int, int], epsilon=1e-6):
    n_tgt, n_src = attn.shape
    src_x, tgt_x = cross_over
    assert 0 < tgt_x < n_tgt
    assert 0 < src_x < n_src
    good_mass, bad_mass = 0, 0
    for ti, row in enumerate(attn):
        assert abs(row.sum() - 1) < epsilon  # row sums to 1
        part1, part2 = row[:src_x], row[src_x:]  # upto src_x is first sentence; after is second sentence
        if ti >= tgt_x:
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
"""


def avg_attn_bleed_fast(attn, cross_over: Tuple[int, int], exclude_eos=False):
    n_layers, n_heads, n_ref, n_src = attn.shape
    src_x, tgt_x = cross_over
    attn = attn.view(-1, n_ref, n_src)

    counter = torch.arange(n_src, device=device)
    good_mask = torch.cat(
        [(counter < src_x).repeat(tgt_x, 1), (counter >= src_x).repeat(n_ref - tgt_x, 1)], dim=0
    )
    if exclude_eos:  # dont consider EOS, i.e. last token, os bleeding
        good_mask[:, -1] = True
        good_mask[-1, :] = True
    # good_score = attn.masked_fill(good_mask, 0).sum()
    # bleed_score = attn.masked_fill(~good_mask, 0).sum()
    good_score = attn.masked_select(good_mask).sum()
    bleed_score = attn.masked_select(~good_mask).sum()
    total_mass = attn.shape[0] * n_ref * 1.0
    assert abs(good_score + bleed_score - total_mass) < (1e-6 * total_mass)
    return (bleed_score / total_mass).item()


def corpus_bleed_rate(batches, decoder: Decoder, output: TextIO, exclude_eos=False):
    rates = []
    for srcs, refs, src_idxs, ref_idxs in batches:
        batch_size = len(srcs)
        assert batch_size == len(refs) == len(src_idxs), len(ref_idxs)
        attns, (src_seq, ref_seq) = get_attns(decoder=decoder, srcs=srcs, refs=refs)
        enc_attn, dec_attn, xattn = attns
        for i in range(batch_size):
            src_len, ref_len = len(src_seq[i]), len(ref_seq[i]) - 1
            i_xttn = xattn[i][:, :, :ref_len, :src_len]  # exclude pads
            # rate = avg_attn_bleed(i_xttn, (src_idxs[i], ref_idxs[i]))
            rate = avg_attn_bleed_fast(i_xttn, (src_idxs[i], ref_idxs[i]), exclude_eos=exclude_eos)
            output.write(f'{rate:.4f}\n')
            rates.append(rate)
    # mean over the corpus
    return sum(rates) / len(rates)


def compute_bleed(exp, **cli_args):
    dec_args = exp.config.get('decoder') or exp.config['tester'].get('decoder', {})
    src_file: TextIO = cli_args.pop('src')
    ref_file: TextIO = cli_args.pop('ref')
    batch_size: int = cli_args.pop('batch_size', 10)
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

    recs = _prep_input()
    batches = make_batches(recs, batch_size=batch_size)
    bleed_rate = corpus_bleed_rate(batches, decoder, output, exclude_eos=True)
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
    print(f'AverageBleed: {rate:.4f}')


if __name__ == '__main__':
    main()
