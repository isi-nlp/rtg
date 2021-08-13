import time
import traceback
from io import StringIO
from typing import List, Tuple, Type, Dict, Any, Optional, Iterator
from pathlib import Path
import math
from dataclasses import dataclass, field
import warnings
import sys
import os

import torch
from torch import nn as nn

from rtg import TranslationExperiment as Experiment
from rtg import log, device, my_tensor as tensor, debug_mode
from rtg.module.generator import GeneratorFactory
from rtg.data.dataset import Field
from rtg.registry import factories, generators

Hypothesis = Tuple[float, List[int]]
StrHypothesis = Tuple[float, str]

if not sys.warnoptions:
    warnings.simplefilter("default") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses


def load_models(models: List[Path], exp: Experiment):
    res = []
    for i, model_path in enumerate(models):
        assert model_path.exists()
        log.info(f"Load Model {i}: {model_path} ")
        chkpt = torch.load(str(model_path), map_location=device)
        model = exp.load_model_with_state(chkpt)
        res.append(model)
    return res


class ReloadEvent(Exception):
    """An exception to reload model with new path
    -- Its a kind of hack to pass event back to caller and redo interactive shell--
    """

    def __init__(self, model_paths, state: Dict[str, Any]):
        super().__init__()
        self.model_paths = model_paths
        self.state = state


@dataclass
class DecoderBatch:

    idxs: List[int] = field(default_factory=list)  # index in the file, for restoring the order
    srcs: List[str] = field(default_factory=list)
    seqs: List[str] = field(default_factory=list)  # processed srcs
    refs: List[str] = field(default_factory=list)  # references for logging if they exist
    ids: List[str] = field(default_factory=list)   # original id column; not to be confused with
    line_count = 0
    tok_count = 0
    max_len = 0
    max_len_buffer = 0   # Some extra buffer for target size; eg: tgt_len = 50 + src_len

    def add(self, idx, src, ref, seq, id):
        self.idxs.append(idx)
        self.srcs.append(src)
        self.refs.append(ref)
        self.seqs.append(seq)
        self.ids.append(id)
        self.line_count += 1
        self.tok_count += len(seq)
        self.max_len = max(self.max_len, len(seq))

    @property
    def padded_tok_count(self):
        return ( self.max_len + self.max_len_buffer ) * self.line_count

    def as_tensors(self, device):
        seqs = torch.zeros(self.line_count, self.max_len, device=device,
                           dtype=torch.long)
        lens = torch.zeros(self.line_count, device=device, dtype=torch.long)
        for i, seq in enumerate(self.seqs):
            seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            lens[i] = len(seq)
        return seqs, lens

    @classmethod
    def from_lines(cls, lines: Iterator[str], batch_size: int, vocab: Field, sort=True,
                   max_src_len=0, max_len_buffer=0):
        """
        Note: this changes the order based on sequence length if sort=True
        :param lines: stream of input lines
        :param batch_size: number of tokens in batch
        :param vocab: Field to use for mapping word pieces to ids
        :param sort: sort based on descending order of length
        :param max_src_len : truncate at length ; 0 disables this
        :return: stream of DecoderBatches
        """
        log.info("Tokenizing sequences")
        buffer = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                log.warning(f"line {i + 1} was empty. inserting a dot (.). "
                            f"Empty lines are problematic when you want line-by-line alignment...")
                line = "."
            cols = line.split('\t')
            id, ref = None, None
            if len(cols) == 1: # SRC
                src = cols[0]
            elif len(cols) == 2: # ID \t SRC
                id, src = cols
            else: # ID \t SRC \t REF
                id, src, ref = cols[:3]
            seq = vocab.encode_as_ids(src, add_eos=True, add_bos=False)
            if max_src_len > 0 and len(seq) > max_src_len:
                log.warning(f"Line {i} full length={len(seq)} ; truncated to {max_src_len}")
                seq = seq[:max_src_len]
            buffer.append((i, src, ref, seq, id))  # idx, src, ref, seq, id

        if sort:
            log.info(f"Sorting based on the length. total = {len(buffer)}")
            buffer = sorted(buffer, reverse=True, key=lambda x: len(x[3]))  # sort by length of seq

        batch = cls()
        batch.max_len_buffer = max_len_buffer
        for idx, src, ref, seq, _id in buffer:
            batch.add(idx=idx, src=src, ref=ref, seq=seq, id=_id)
            if batch.padded_tok_count >= batch_size:
                yield batch
                batch = cls()
                batch.max_len_buffer = max_len_buffer

        if batch.line_count > 0:
            yield batch


class Decoder:
    default_beam_size = 5

    def __init__(self, model, gen_factory: Type[GeneratorFactory], exp: Experiment, gen_args=None,
                 debug=debug_mode):
        self.model = model
        self.exp = exp
        self.gen_factory = gen_factory
        self.debug = debug
        self.gen_args = gen_args if gen_args is not None else {}
        self.pad_val = exp.tgt_vocab.pad_idx
        self.bos_val = exp.tgt_vocab.bos_idx
        self.eos_val = exp.tgt_vocab.eos_idx

        self.dec_bos_cut = self.exp.config.get('trainer', {}).get('dec_bos_cut', False)
        (log.info if self.dec_bos_cut else log.debug)(f"dec_bos_cut={self.dec_bos_cut}")

    def generator(self, x_seqs, x_lens):
        return self.gen_factory(self.model, field=self.exp.tgt_vocab,
                                x_seqs=x_seqs, x_lens=x_lens, **self.gen_args)

    @classmethod
    def combo_new(cls, exp: Experiment, model_paths: List[str], weights: List[float]):
        assert len(model_paths) == len(weights), 'one weight per model needed'
        assert abs(sum(weights) - 1) < 1e-3, 'weights must sum to 1'
        log.info(f"Combo mode of {len(model_paths)} models :\n {list(zip(model_paths, weights))}")
        model_paths = [Path(m) for m in model_paths]
        models = load_models(model_paths, exp)
        from rtg.syscomb import Combo
        combo = Combo(models)
        return cls.new(exp, model=combo, model_type='combo')

    @classmethod
    def new(cls, exp: Experiment, model=None, gen_args=None,
            model_paths: Optional[List[str]] = None,
            ensemble: int = 1, model_type: Optional[str] = None):
        """
        create a new decoder
        :param exp: experiment
        :param model: Optional pre initialized model
        :param gen_args: any optional args needed for generator
        :param model_paths: optional model paths
        :param ensemble: number of models to use for ensembling (if model is not specified)
        :param model_type: model_type ; when not specified, model_type will be read from experiment
        :return:
        """
        if not model_type:
            model_type = exp.model_type
        if model is None:
            factory = factories[model_type]
            model = factory(exp=exp, **exp.model_args)[0]
            state = exp.maybe_ensemble_state(model_paths=model_paths, ensemble=ensemble)
            model.load_state_dict(state)
            log.info("Successfully restored the model state.")
        elif isinstance(model, nn.DataParallel):
            model = model.module

        model = model.eval().to(device=device)
        generator = generators[model_type]
        if exp.optim_args[1] and exp.optim_args[1].get('criterion') == 'binary_cross_entropy':
            log.info("((Going to decode in multi-label mode))")
            gen_args = gen_args or {}
            gen_args['multi_label'] = True
        return cls(model, generator, exp, gen_args)

    def greedy_decode(self, x_seqs, x_lens, max_len, **args) -> List[Hypothesis]:
        """
        Implements a simple greedy decoder
        :param x_seqs:
        :param x_lens: length of x sequences
        :param max_len:
        :return:
        """
        device = x_seqs.device
        batch_size = x_seqs.size(0)
        if self.dec_bos_cut:
            ys = x_seqs[:, :1]
            x_seqs = x_seqs[:, 1:]
            x_lens -= 1
        else:
            ys = torch.full(size=(batch_size, 1), fill_value=self.bos_val, dtype=torch.long,
                            device=device)
        gen = self.generator(x_seqs, x_lens)
        scores = torch.zeros(batch_size, device=device)
        actives = ys[:, -1] != self.eos_val
        max_x_len = x_lens.max().item()
        for i in range(1, max_x_len + max_len + 1):
            if actives.sum() == 0:  # all sequences Ended
                break
            log_prob = gen.generate_next(ys)
            max_prob, next_word = torch.max(log_prob, dim=1)
            scores += max_prob
            ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
            actives &= ys[:, -1] != self.eos_val

        result = []
        for i in range(batch_size):
            result.append((scores[i].item(), ys[i, 1:].tolist()))
        return result

    @staticmethod
    def masked_select(x, mask):
        assert x.shape[0] == mask.shape[0]
        assert mask.shape[1] == 1
        selected = x.masked_select(mask)
        return selected.view(-1, x.size(1))

    @staticmethod
    def repeat_adjacent(x, n, dim=0):
        """
        repeat along a dimension values are adjacent.
        unlike torch.Tensor.repeat() which repeats at the end instead of adjacent.
        :param x: input tensor
        :param n: how many times to repeat
        :param dim: which dimension
        :return: repeated tensor
        """
        shape = list(x.shape)
        # add a new dimension and expand it beam size
        expand_shape = [-1] * (len(x.shape) + 1)
        expand_shape[dim + 1] = n
        x = x.unsqueeze(dim + 1).expand(*expand_shape)
        shape[dim] *= n  # reduce to original num of dims but given dim has n times more
        return x.contiguous().view(*shape)

    def beam_decode(self, x_seqs, x_lens, max_len, beam_size=default_beam_size, num_hyp=1,
                    lp_alpha: float = 0., **args) -> List[List[Hypothesis]]:
        """
        Beam decoder
        :param x_seqs: input x_seqs as a padded tensor
        :param x_lens: lengths of x_lengths
        :param max_len: maximum time steps to run
        :param beam_size: how many beams
        :param num_hyp: how many hypothesis to return ( must be <= beam_size)
        :param lp_alpha: length penalty (0.0 means disables)
        :return:
        """
        args = dict((k, v) for k, v in args.items() if v is not None)
        if args:
            warnings.warn(f"Ignored args: {args}. To remove this message simply remove the args")
        assert beam_size >= num_hyp
        device = x_seqs.device
        batch_size = x_seqs.size(0)
        # ys = torch.zeros(batch_size, beam_size, max_len + 1, dtype=torch.long, device=device)
        if self.dec_bos_cut:
            # cut first time step from xs and repeat beam_size times, add time dim
            # [Batch] -> [Batch x 1=Beam] -> [Batch x Beam]  -> [Batch x Beam x 1=Time]
            ys = x_seqs[:, 0].unsqueeze(-1).expand(-1, beam_size).unsqueeze(-1)
            x_seqs = x_seqs[:, 1:]  # cut
            x_lens -= 1
        else:
            ys = torch.full((batch_size, beam_size, 1), fill_value=self.bos_val,
                            device=device, dtype=torch.long)

        # repeat x_seqs and x_lens beam times
        beamed_x_seqs = self.repeat_adjacent(x_seqs, n=beam_size, dim=0)
        beamed_x_lens = self.repeat_adjacent(x_lens, n=beam_size, dim=0)
        gen = self.generator(beamed_x_seqs, beamed_x_lens)
        scores = torch.zeros(batch_size, beam_size, device=device)
        actives = ys[:, :, 0] != self.eos_val
        lengths = torch.full((batch_size, beam_size), fill_value=max_len, device=device,
                             dtype=torch.long)
        max_x_len = x_lens.max().item()
        for t in range(1, max_x_len + max_len + 1):
            if actives.sum() == 0:  # all sequences Ended
                break
            # [Batch x Beams x Vocab] <-- [Batch x Beams x Time]
            flat_ys = ys.contiguous().view(batch_size * beam_size, -1)
            log_prob = gen.generate_next(flat_ys)  # ys upto current time step
            log_prob = log_prob.view(batch_size, beam_size, -1)

            if t == 1:
                # Note: since sequences are duplicated, to start with
                # we need to pick the top k beams from a single beam
                # How? mask out all beams, except the first beam
                beam_mask = torch.full((batch_size, beam_size, 1), fill_value=1, device=device,
                                       dtype=torch.bool)
                beam_mask[:, 0, :] = 0
                log_prob.masked_fill_(mask=beam_mask, value=float('-inf'))

            inactives = ~actives
            if inactives.sum() > 0:
                # Goal: do not let the inactive beams grow. How? this is tricky
                # we set -inf to all next words of inactive beams (so none of them make to topk)
                log_prob.masked_fill_(mask=inactives.unsqueeze(-1), value=float('-inf'))
                # But we need to preserve the inactive beam (just one copy) if it is still in topk. how?
                # just set zero to just one word of inactive beam
                # shouldn't matter which word since an EOS has already appeared --> pick index 0 word
                log_prob[:, :, 0].masked_fill_(mask=inactives, value=0.0)

            # add current beam_scores all possible next_words
            # broadcast scores to each word in vocab [Batch x Beams x Vocab=1]
            next_scores = scores.unsqueeze(-1) + log_prob

            # max_probs and next_words: [Batch x Beams x Beams] --> [Batch x Beams*Beams]
            next_scores, next_words = next_scores.topk(k=beam_size, dim=-1, largest=True)
            next_scores = next_scores.view(batch_size, beam_size * beam_size)
            next_words = next_words.view(batch_size, beam_size * beam_size)

            # Trim beams: [Batch, Beams] <-- [Batch, Beams*Beams]
            scores, next_words_idxs = next_scores.topk(k=beam_size, dim=-1, largest=True)
            next_words = next_words.gather(dim=1, index=next_words_idxs)

            # task: rearrange ys based on the newer ranking of beams
            # ys_idx: [Beams] --> [Beams x 1] --> [Beams x Beams]
            #          --> [1 x Beams x Beams] --> [Batch x Beams * Beams]
            ys_idx = torch.arange(beam_size, device=device) \
                .unsqueeze(-1).expand(-1, beam_size) \
                .unsqueeze(0).expand(batch_size, -1, -1).contiguous() \
                .view(batch_size, beam_size * beam_size)
            # [Batch x Beams] <- [Batch x Beams*Beams] as per the topk next_scores of beams
            ys_idx = ys_idx.gather(dim=1, index=next_words_idxs)
            ys_idx = ys_idx.unsqueeze(-1).expand_as(ys)  # expand along time dim
            ys = ys.gather(1, ys_idx)  # re arrange beams
            ys = torch.cat([ys, next_words.unsqueeze(-1)], dim=-1)  # cat along the time dim

            # Task: update lengths and active flag of beam
            ended_beams = actives & (next_words == self.eos_val)  # it was active but saw EOS now @t
            lengths.masked_fill_(mask=ended_beams, value=t)
            actives &= next_words != self.eos_val  # was active and not EOS yet

        ys = ys[:, :, 1:]  # remove BOS
        if lp_alpha > 0:
            # Page 12 of Wu et al (2016) Google NMT : https://arxiv.org/pdf/1609.08144.pdf
            # score(y, X) = \frac{ logP(Y | X) }{ lp(Y)}
            # lp(Y) = \frac{ (5 + |Y|)^α }{ (5 + 1)^α }
            penalty = (5 + lengths.float()).pow(lp_alpha) / math.pow(6, lp_alpha)
            scores = scores / penalty
        n_hyp_scores, n_hyp_idxs = scores.topk(k=num_hyp, dim=-1)  # pick num_hyp beams
        result = []
        for seq_idx in range(batch_size):
            result.append([])
            for hyp_score, beam_idx in zip(n_hyp_scores[seq_idx], n_hyp_idxs[seq_idx]):
                result[-1].append((hyp_score, ys[seq_idx, beam_idx, :].tolist()))
        return result

    @property
    def inp_vocab(self) -> Field:
        # the choice of vocabulary can be tricky, because of bidirectional model
        if self.exp.model_type == 'binmt':
            return {
                'E1': self.exp.src_vocab,
                'E2': self.exp.tgt_vocab
            }[self.gen_args['path'][:2]]
        else:  # all others go from source as input to target as output
            return self.exp.src_vocab

    @property
    def out_vocab(self) -> Field:
        # the choice of vocabulary can be tricky, because of bidirectional model
        if self.exp.model_type == 'binmt':
            return {
                'D1': self.exp.src_vocab,
                'D2': self.exp.tgt_vocab
            }[self.gen_args['path'][-2:]]
        else:  # all others go from source as input to target as output
            return self.exp.tgt_vocab

    def decode_sentence(self, line: str, max_len=20, prepared=False, **args) -> List[StrHypothesis]:

        line = line.strip()
        if prepared:
            in_seq = [int(t) for t in line.split()]
            if in_seq[0] != self.bos_val:
                in_seq.insert(0, self.bos_val)
            if in_seq[-1] != self.eos_val:
                in_seq.append(self.eos_val)
        else:
            in_seq = self.inp_vocab.encode_as_ids(line, add_eos=True, add_bos=False)
        in_seqs = tensor(in_seq, dtype=torch.long).view(1, -1)
        in_lens = tensor([len(in_seq)], dtype=torch.long)
        if self.debug:
            greedy_score, greedy_out = self.greedy_decode(in_seqs, in_lens, max_len, **args)[0]
            greedy_out = self.out_vocab.decode_ids(greedy_out, trunc_eos=True)
            log.debug(f'Greedy : score: {greedy_score:.4f} :: {greedy_out}')

        beams: List[List[Hypothesis]] = self.beam_decode(in_seqs, in_lens, max_len, **args)
        beams = beams[0]  # first sentence, the only one we passed to it as input
        result = []
        for i, (score, beam_toks) in enumerate(beams):
            out = self.out_vocab.decode_ids(beam_toks, trunc_eos=True)
            if self.debug:
                log.debug(f"Beam {i}: score:{score:.4f} :: {out}")
            result.append((score, out))
        return result

    def next_word_distr(self, past_seq, x_seqs=None, x_lens=None):
        """
        Gets log distribution of next word
        :param past_seq: paste sequence
        :param x_seqs: optional; source sequence,
        :param x_lens: optional; source sequence length
        :return: log probability distribution of next word
        """
        return self.generator(x_seqs=x_seqs, x_lens=x_lens).generate_next(past_seq)

    # noinspection PyUnresolvedReferences
    def decode_interactive(self, **args):
        import sys
        import readline
        helps = [(':quit', 'Exit'),
                 (':help', 'Print this help message'),
                 (':beam_size <n>', 'Set beam size to n'),
                 (':lp_alpha <n>', 'Set length penalty alpha'),
                 (':num_hyp <k>', 'Print top k hypotheses'),
                 (':debug', 'Flip debug flag'),
                 (':models', 'show all available models of this experiment'),
                 (':model <number>', 'reload shell with the model chosen by <number>')
                 ]
        if self.exp.model_type == 'binmt':
            helps.append((':path <path>', 'BiNMT modules: {E1D1, E2D2, E1D2E2D1, E2D2E1D2}'))

        def print_cmds():
            for cmd, msg in helps:
                print(f"\t{cmd:15}\t-\t{msg}")

        global debug_mode
        print("Launching Interactive shell...")
        import rtg.module.generator as gen
        gen.INTERACTIVE = True
        print_cmds()
        print_state = True
        while True:
            if print_state:
                state = '  '.join(f'{k}={v}' for k, v in args.items())
                if self.exp.model_type == 'binmt':
                    state += f'  path={self.gen_args.get("path")}'
                state += f'  debug={debug_mode}'
                print('\t|' + state)
            print_state = False
            line = input('Input: ')
            line = line.strip()
            if not line:
                continue
            try:
                if line == ':quit':
                    break
                elif line == ':help':
                    print_cmds()
                elif line.startswith(":beam_size"):
                    args['beam_size'] = int(line.replace(':beam_size', '').replace('=', '').strip())
                    print_state = True
                elif line.startswith(":num_hyp"):
                    args['num_hyp'] = int(line.replace(':num_hyp', '').replace('=', '').strip())
                    print_state = True
                elif line.startswith(":lp_alpha"):
                    args['lp_alpha'] = float(line.replace(':lp_alpha', '').replace('=', '').strip())
                    print_state = True
                elif line == ":debug":
                    debug_mode = self.debug = not debug_mode

                    print_state = True
                elif line.startswith(":path"):
                    self.gen_args['path'] = line.replace(':path', '').replace('=', '').strip()
                    print_state = True
                elif line.startswith(":models"):
                    for i, mod_path in enumerate(self.exp.list_models()):
                        print(f"\t{i}\t{mod_path}")
                elif line.startswith(":model"):
                    mod_idxs = [int(x) for x in
                                line.replace(":model", "").replace("=", "").strip().split()]
                    models = self.exp.list_models()
                    mod_paths = []
                    for mod_idx in mod_idxs:
                        if 0 <= mod_idx < len(models):
                            mod_paths.append(str(models[mod_idx]))
                        else:
                            print(f"\tERROR: Index {mod_idx} is invalid")
                    if mod_paths:
                        print(f"\t Switching to models {mod_paths}")
                        raise ReloadEvent(mod_paths, state=args)
                else:
                    start = time.time()
                    res = self.decode_sentence(line, **args)
                    print(f'\t|took={1000 * (time.time() - start):.3f}ms')
                    for score, hyp in res:
                        print(f'  {score:.4f}\t{hyp}')
            except ReloadEvent as re:
                raise re  # send it to caller
            except EOFError as e1:
                break
            except Exception:
                traceback.print_exc()
                print_state = True

    @staticmethod
    def _remove_null_vals(args: Dict):
        return {k: v for k, v in args.items() if v is not None}  # remove None args

    def decode_file(self, inp: Iterator[str], out: StringIO,
                    num_hyp=1, batch_size=1, max_src_len=-1, **args):
        args = self._remove_null_vals(args)
        log.info(f"Args to decoder : {args} and num_hyp={num_hyp} "
                 f"batch_size={batch_size} max_src_len={max_src_len}")

        batches: Iterator[DecoderBatch] = DecoderBatch.from_lines(
            inp, batch_size=batch_size, vocab=self.inp_vocab, max_src_len=max_src_len,
            max_len_buffer=args.get('max_len', 1))

        def _decode_all():
            buffer = []
            for batch in batches:
                in_seqs, in_lens = batch.as_tensors(device=device)
                batched_hyps: List[List[Hypothesis]] = self.beam_decode(in_seqs, in_lens,
                                                                        num_hyp=num_hyp, **args)
                assert len(batched_hyps) == batch.line_count
                for i, hyps in enumerate(batched_hyps):
                    idx = batch.idxs[i]
                    src = batch.srcs[i]
                    _id = batch.ids[i]
                    log.info(f"{idx}: SRC: {batch.srcs[i]}")
                    ref = batch.refs[i]  # just for the sake of logging, if it exists
                    if ref:
                        log.info(f"{idx}: REF: {batch.refs[i]}")

                    result = []
                    for j, (score, hyp) in enumerate(hyps):
                        hyp_line = self.out_vocab.decode_ids(hyp,
                                                             trunc_eos=True)  # tok ids to string
                        log.info(f"{idx}: HYP{j}: {score:g} : {hyp_line}")
                        result.append((score, hyp_line))
                    buffer.append((idx, src, result, _id))

            buffer = sorted(buffer, key=lambda x: x[0])  # restore order
            for _, src, result, _id in buffer:
                yield src, result, _id

        streamed_results: Iterator[Tuple[str, List[StrHypothesis], Any]] = _decode_all()
        for src, hyps, _id in streamed_results:
            prefix = f'{_id}\t' if _id else ''  # optional Id
            out_line = '\n'.join(f'{prefix}{hyp}\t{score:.4f}' for score, hyp in hyps)
            out.write(f'{out_line}\n')
            if num_hyp > 1:
                out.write('\n')

    def decode_stream(self, inp: Iterator[str], out: StringIO,
                      max_src_len=-1, **args):
        args = self._remove_null_vals(args)
        log.info(f"Args to decoder : {args} max_src_len={max_src_len}")

        for inp_line in inp:
            log.info(f"SRC: {inp_line}")
            out_line = self.decode_sentence(line=inp_line, **args)[0][1]    # 0th result, 1st hyp
            log.info(f"HYP: {out_line} \n")

            out.write(f'{out_line}\n')
            out.flush()
