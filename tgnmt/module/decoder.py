import torch
from tgnmt import log, device, my_tensor as tensor
from tgnmt.dataprep import BLANK_TOK, BOS_TOK, subsequent_mask
from tgnmt.module.t2t import EncoderDecoder


class Decoder:

    pad_val = BLANK_TOK[1]
    bos_val = BOS_TOK[1]
    default_beam_size = 5

    def __init__(self, model, exp):
        self.exp = exp
        self.model = model
        self.model.eval()

    @classmethod
    def new(cls, exp):
        mod_args = exp.get_model_args()
        check_pt_file, _ = exp.get_last_saved_model()
        log.info(f" Restoring state from {check_pt_file}")
        model = EncoderDecoder.make_model(**mod_args)[0].to(device)
        model.load_state_dict(torch.load(check_pt_file))
        return cls(model, exp)

    def greedy_decode(self, x_seqs, max_len, **args):
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
        for i in range(max_len):
            out = self.model.decode(memory, x_mask, ys, subsequent_mask(ys.size(1)))
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
        return ys[:, 1:]  # exclude BOS

    def init_beam(self, x_seqs, beam_size):

        x_mask = (x_seqs != self.pad_val).unsqueeze(1)
        memory = self.model.encode(x_seqs, x_mask)
        batch_size = x_seqs.size(0)
        assert batch_size == 1, 'Currently doing one sequence at a time. TODO: do full batch'

        # Initialize k beams, with BOS token
        ys = torch.full(size=(batch_size, 1), fill_value=self.bos_val, dtype=torch.long, device=device)
        out = self.model.decode(memory, x_mask, ys, subsequent_mask(ys.size(1)))
        prob = self.model.generator(out[:, -1])  # [batch_size, Vocab]

        top_probs, nxt_idx = torch.topk(prob, k=beam_size, dim=1)  # [batch, beam_size],[batch, beam_size]

        # repeat evrything beam_size times
        expanded_ys = ys.repeat(1, beam_size).view(batch_size * beam_size, -1)
        beamed_ys = torch.cat([expanded_ys, nxt_idx.view(-1, 1)], dim=1)
        beam_scores = top_probs.log().view(batch_size*beam_size, 1)
        memory_size = memory.size()
        beamed_mem = memory.repeat(1, beam_size, 1).view(memory_size[0]*beam_size, memory_size[1], memory_size[2])
        mask_size = x_mask.size()
        beamed_mask = x_mask.repeat(1, beam_size, 1).view(mask_size[0]*beam_size, mask_size[1], mask_size[2])
        return beamed_mem, beamed_mask, beamed_ys, beam_scores

    def beam_decode(self, x_seqs, max_len, beam_size=default_beam_size, **args):
        """Implements beam decoding"""

        memory, x_mask, ys, scores = self.init_beam(x_seqs, beam_size)
        for t in range(1, max_len):
            out = self.model.decode(memory, x_mask, ys, subsequent_mask(ys.size(1)))
            prob = self.model.generator(out[:, -1])  # [batch*beam, Vocab]
            # broad cast scores along row and sum  log probabilities
            next_scores = scores.view(-1, 1) + prob.log()

            top_scores, nxt_idx = torch.topk(next_scores, k=beam_size, dim=1)  # [batch*beam, beam],[batch*beam, beam]
            # Now we got beam_size*beam_size heads, task: shrink it to beam_size
            # Since the ys will change, after re-scoring, we will make a new tensor for new ys
            new_ys = torch.full(size=(ys.size(0), ys.size(1) + 1), fill_value=self.pad_val, device=device,
                                dtype=torch.long)
            for i in range(x_seqs.size(0)):
                # beams of i'th sequence in batch
                start, end = i*beam_size, (i+1) * beam_size
                seqi_scores, seqi_nxt_idx = top_scores[start:end, :], nxt_idx[start:end, :]

                # picking top k out of k*k
                seqi_top_scores, seqi_nxt_idx_idx = seqi_scores.view(-1).topk(k=beam_size)
                # copy the old beams corresponding to current top scores
                scores[start:end, 0] = seqi_top_scores
                beam_idx = seqi_nxt_idx_idx / beam_size
                new_ys[start:end, :-1] = ys[start:end, :].index_select(0, beam_idx)
                # copy the new word indices to last step
                new_ys[start:end, -1] = seqi_nxt_idx.view(-1)[seqi_nxt_idx_idx]

            ys = new_ys
        # TODO: Special treatment for EOS tokens and padding tokens in scoring
        return ys[:, 1:], scores  # exclude BOS

    def decode_sentence(self, line: str, max_len=20, **args) -> str:
        in_toks = line.strip().split()
        in_seq = self.exp.src_field.seq2idx(in_toks, add_bos=True, add_eos=True)
        in_seqs = tensor(in_seq, dtype=torch.long).view(1, -1)

        greedy_out = self.greedy_decode(in_seqs, max_len, **args)[0]
        greedy_toks = self.exp.tgt_field.idx2seq(greedy_out, trunc_eos=True)
        greedy_out = ' '.join(greedy_toks)
        log.info(f'Greedy : {greedy_out}')

        beam_outs, scores = self.beam_decode(in_seqs, max_len, **args)
        for i in range(beam_outs.size(0)):
            out = ' '.join(self.exp.tgt_field.idx2seq(beam_outs[i], trunc_eos=True))
            log.info(f"Beam {i}: score:{scores[i]} :: {out}")
        return greedy_out

    def decode_file(self, inp, out, **args):
        for i, line in enumerate(inp):
            line = line.strip()
            if not line:
                log.warning(f"line {i+1} was empty. skipping it for now. "
                            f"Empty lines are problematic when you want line-by-line alignment...")
                continue
            cols = line.split('\t')
            input = cols[0]
            log.info(f"INP: {i}: {cols[0]}")
            if len(cols) > 1:  # assumption: second column is reference
                log.info(f"REF: {i}: {cols[1]}")
            out_line = self.decode_sentence(input, **args)
            out.write(f'{out_line}\n')
            log.info(f"OUT: {i}: {out_line}\n")

