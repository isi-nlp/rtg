import torch
from tgnmt import log, device, my_tensor as tensor
from tgnmt.dataprep import BLANK_TOK, BOS_TOK, EOS_TOK, subsequent_mask
from tgnmt.module.t2t import EncoderDecoder
from typing import List, Tuple

Hypothesis = Tuple[float, List[int]]
StrHypothesis = Tuple[float, str]


class Decoder:

    pad_val = BLANK_TOK[1]
    bos_val = BOS_TOK[1]
    eos_val = EOS_TOK[1]
    default_beam_size = 5

    def __init__(self, model, exp, debug=False):
        self.exp = exp
        self.model = model
        self.model.eval()
        self.debug = debug

    @classmethod
    def new(cls, exp):
        mod_args = exp.get_model_args()
        check_pt_file, _ = exp.get_last_saved_model()
        log.info(f" Restoring state from {check_pt_file}")
        model = EncoderDecoder.make_model(**mod_args)[0].to(device)
        model.load_state_dict(torch.load(check_pt_file))
        return cls(model, exp)

    def greedy_decode(self, x_seqs, max_len, **args) -> List[Hypothesis]:
        """
        Implements a simple greedy decoder
        :param x_seqs:
        :param max_len:
        :return:
        """
        x_mask = (x_seqs != self.pad_val).unsqueeze(1)
        memory = self.model.encode(x_seqs, x_mask)
        batch_size = x_seqs.size(0)
        ys = torch.full(size=(batch_size, 1), fill_value=self.bos_val, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size, device=device)

        actives = ys[:, -1] != self.eos_val
        for i in range(max_len):
            if actives.sum() == 0:  # all sequences Ended
                break
            out = self.model.decode(memory, x_mask, ys, subsequent_mask(ys.size(1)))
            log_prob = self.model.generator(out[:, -1])
            max_prob, next_word = torch.max(log_prob, dim=1)
            scores += max_prob
            ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
            actives &= ys[:, -1] != self.eos_val

        result = []
        for i in range(batch_size):
            result.append((scores[i].item(), ys[i, 1:]))
        return result

    @staticmethod
    def masked_select(x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.shape[1] == 1
        selected = x.masked_select(mask)
        return selected.view(-1, x.size(1))

    def beam_decode(self, x_seqs, max_len, beam_size=default_beam_size, num_hyp=None, **args) -> List[List[Hypothesis]]:
        """

        :param x_seqs: input batch of sequences
        :param max_len:  maximum length to consider if decoder doesnt produce EOS token
        :param beam_size: beam size
        :param num_hyp: number of hypothesis in each beam to return
        :param args:
        :return: List of num_hyp Hypothesis for each sequence in the batch.
         Each hypothesis consists of score and a list of word indexes.
        """
        """Implements beam decoding"""
        # repeat beam size
        batch_size = x_seqs.size(0)
        assert batch_size == 1  # TODO: test large batches
        if not num_hyp:
            num_hyp = beam_size
        beam_size = max(beam_size, num_hyp)

        # Everything beamed_*  below is the batch repeated beam_size times
        beamed_batch_size = batch_size * beam_size
        beamed_x_seqs = x_seqs.repeat(1, beam_size).view(beamed_batch_size, -1)
        beamed_x_mask = (beamed_x_seqs != self.pad_val).unsqueeze(1)
        beamed_memory = self.model.encode(beamed_x_seqs, beamed_x_mask)

        beamed_ys = torch.full(size=(beamed_batch_size, 1), fill_value=self.bos_val, dtype=torch.long, device=device)
        beamed_scores = torch.zeros((beamed_batch_size, 1), device=device)

        beam_active = torch.ones((beamed_batch_size, 1), dtype=torch.uint8, device=device)
        # zeros means ended, one means active
        for t in range(1, max_len):
            if beam_active.sum() == 0:
                break
            out = self.model.decode(beamed_memory, beamed_x_mask, beamed_ys, subsequent_mask(beamed_ys.size(1)))
            log_prob = self.model.generator(out[:, -1])  # [batch*beam, Vocab]
            # broad cast scores along row (broadcast) and sum  log probabilities

            next_scores = beamed_scores + beam_active.float() * log_prob  # Zero out inactive beams

            top_scores, nxt_idx = next_scores.topk(k=beam_size, dim=-1)  # [batch*beam, beam],[batch*beam, beam]
            # Now we got beam_size*beam_size heads, task: shrink it to beam_size
            # Since the ys will change, after re-scoring, we will make a new tensor for new ys
            new_ys = torch.full(size=(beamed_batch_size, beamed_ys.size(1) + 1), fill_value=self.pad_val,
                                device=device, dtype=torch.long)

            for i in range(batch_size):
                # going to picking top k out of k*k beams for each sequence in batch
                # beams of i'th sequence in batch have this start and end
                start, end = i * beam_size, (i+1) * beam_size
                if beam_active[start:end].sum() == 0:
                    # current sequence ended
                    new_ys[:start:end, :-1] = beamed_ys[start:end]
                    continue

                if t == 1:
                    # initialization ; since sequences are duplicated in the start, we just pick the first row
                    # it is must, otherwise, beam will select the top 1 of each beam which is same but duplicated
                    seqi_top_scores, seqi_nxt_ys = top_scores[start, :], nxt_idx[start, :]
                    # ys seen so far remains same; no reordering
                    new_ys[start:end, :-1] = beamed_ys[start:end]
                    seqi_top_scores = seqi_top_scores.view(-1, 1)
                else:
                    seqi_nxt_ys = torch.full((beam_size,1), fill_value=self.pad_val, device=device)
                    seqi_top_scores = torch.zeros((beam_size, 1), device=device)

                    # ignore the inactive beams, don't grow them any further
                    # INACTIVE BEAMS: Preserve the inactive beams, just copy them
                    seqi_inactive_mask = (beam_active[start:end, -1] == 0).view(-1, 1)
                    seqi_inactive_count = seqi_inactive_mask.sum()
                    active_start = start + seqi_inactive_count    # [start, ... active_start-1, active_start, ... end]
                    if seqi_inactive_count > 0: # if there are some inactive beams
                        seqi_inactive_ys = self.masked_select(beamed_ys[start:end, :], seqi_inactive_mask)
                        new_ys[start: active_start, :-1] = seqi_inactive_ys  # Copy inactive beams
                        seqi_top_scores[start:active_start, :] = \
                            self.masked_select(beamed_scores[start:end, :], seqi_inactive_mask)

                    # ACTIVE BEAMS: the remaining beams should be let to grow
                    seqi_active_mask = (beam_active[start:end, -1] == 1).view(-1, 1)
                    seqi_active_count = seqi_active_mask.sum()  # task is to select these many top scoring beams

                    seqi_scores = self.masked_select(top_scores[start:end, :], seqi_active_mask)
                    seqi_nxt_idx = self.masked_select(nxt_idx[start:end, :], seqi_active_mask)

                    seqi_active_top_scores, seqi_nxt_idx_idx = seqi_scores.view(-1).topk(k=seqi_active_count)
                    seqi_top_scores[active_start:end, 0] = seqi_active_top_scores
                    seqi_nxt_ys[active_start: end, 0] = seqi_nxt_idx.view(-1)[seqi_nxt_idx_idx]

                    # Select active ys
                    active_beam_idx = seqi_nxt_idx_idx / beam_size
                    seqi_active_ys = self.masked_select(beamed_ys[start:end, :], seqi_active_mask)
                    seqi_active_old_ys = seqi_active_ys.index_select(0, active_beam_idx)
                    new_ys[active_start:end, :-1] = seqi_active_old_ys

                    # Update status of beam_active flags
                    beam_active[active_start:end, :] = self.masked_select(beam_active[start:end, :], seqi_active_mask) \
                        .index_select(0, active_beam_idx)
                    if active_start > start:
                        beam_active[start:active_start, -1] = 0  # inactive beams are set to zero

                beamed_scores[start:end, :] = seqi_top_scores
                # copy the new word indices to last step
                new_ys[start:end, -1] = seqi_nxt_ys.view(-1)
            beamed_ys = new_ys
            # AND to update active flag
            beam_active = beam_active & (beamed_ys[:, -1] != self.eos_val).view(beamed_batch_size, 1)

        result = []
        # reverse sort based on the score
        for i in range(batch_size):
            result.append([])
            start, end = i * beam_size, (i + 1) * beam_size
            scores, indices = beamed_scores[start:end, :].view(-1).sort(descending=True)
            for j in range(beam_size):
                if len(result[-1]) == num_hyp:
                    continue
                result[-1].append((scores[j].item(), beamed_ys[start+indices[j], 1:].squeeze()))
        return result

    def decode_sentence(self, line: str, max_len=20, prepared=False, **args) -> List[StrHypothesis]:
        in_toks = line.strip().split()
        if prepared:
            in_seq = [int(t) for t in in_toks]
            if in_seq[0] != self.bos_val:
                in_seq.insert(0, self.bos_val)
            if in_seq[-1] != self.eos_val:
                in_seq.append(self.eos_val)
        else:
            in_seq = self.exp.src_field.seq2idx(in_toks, add_bos=True, add_eos=True)
        in_seqs = tensor(in_seq, dtype=torch.long).view(1, -1)
        if self.debug:
            greedy_score, greedy_out = self.greedy_decode(in_seqs, max_len, **args)[0]
            greedy_toks = self.exp.tgt_field.idx2seq(greedy_out, trunc_eos=True)
            greedy_out = ' '.join(greedy_toks)
            log.debug(f'Greedy : score: {greedy_score} :: {greedy_out}')

        beams: List[List[Hypothesis]] = self.beam_decode(in_seqs, max_len, **args)
        beams = beams[0]  # first sentence, the only one we passed to it as input
        result = []
        for i, (score, beam_toks) in enumerate(beams):
            out = ' '.join(self.exp.tgt_field.idx2seq(beam_toks, trunc_eos=True))
            log.debug(f"Beam {i}: score:{score} :: {out}")
            result.append((score, out))
        return result

    def decode_file(self, inp, out, **args):
        for i, line in enumerate(inp):
            line = line.strip()
            if not line:
                log.warning(f"line {i+1} was empty. skipping it for now. "
                            f"Empty lines are problematic when you want line-by-line alignment...")
                continue
            cols = line.split('\t')
            input = cols[0]
            log.debug(f"INP: {i}: {cols[0]}")
            if len(cols) > 1:  # assumption: second column is reference
                log.debug(f"REF: {i}: {cols[1]}")
            result = self.decode_sentence(input, **args)
            num_hyp = args['num_hyp']
            out_line = '\n'.join(f'{hyp}\t{score:.4f}' for score, hyp in result)
            out.write(f'{out_line}\n')
            log.debug(f"OUT: {i}: {out_line}\n")
            if num_hyp > 1:
                out.write('\n')

