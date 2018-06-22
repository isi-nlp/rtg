from typing import Dict, Callable

import torch
from tgnmt import log, device, my_tensor as tensor
from tgnmt.dataprep import BLANK_TOK, BOS_TOK, subsequent_mask
from tgnmt import TranslationExperiment as Experiment


class GreedyDecoder:

    pad_val = BLANK_TOK[1]
    bos_val = BOS_TOK[1]

    def __init__(self, exp: Experiment, factory: Callable, args: Dict, check_pt_file: str):
        self.exp = exp
        self.model, _ = factory(**args)
        log.info(f" Restoring state from {check_pt_file}")
        self.model.load_state_dict(torch.load(check_pt_file))
        self.model = self.model.to(device)
        self.model.eval()  # turn off training mode

    def greedy_decode(self, x_seqs: torch.Tensor, max_len: int):
        # TODO: batch decode
        x_mask = (x_seqs != self.pad_val).unsqueeze(1)
        memory = self.model.encode(x_seqs, x_mask)
        batch_size = x_seqs.size(0)
        ys = torch.full(size=(batch_size, 1), fill_value=self.bos_val, dtype=torch.long, device=device)
        for i in range(max_len - 1):
            out = self.model.decode(memory, x_mask, ys, subsequent_mask(ys.size(1)))
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
        return ys[:, 1:]  # exclude BOS

    def decode_sentence(self, line: str, max_len=20) -> str:
        in_toks = line.strip().split()
        in_seq = self.exp.src_field.seq2idx(in_toks, add_bos=True, add_eos=True)
        in_seqs = tensor(in_seq, dtype=torch.long).view(1, -1)

        out_seqs = self.greedy_decode(in_seqs, max_len=max_len)
        out_toks = self.exp.tgt_field.idx2seq(out_seqs[0], trunc_eos=True)
        return ' '.join(out_toks)

    def decode_file(self, inp, out):
        for i, line in enumerate(inp):
            line = line.strip()
            log.info(f" Input: {i}: {line}")
            out_line = self.decode_sentence(line)
            log.info(f"Output: {i}: {out_line}\n")
            out.write(f'{out_line}\n')


class GreedyDecoderDev(GreedyDecoder):
    """Same as Greedy decoder, but it accepts pre-initialized model
     instead of attempting to load from serialized file"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
