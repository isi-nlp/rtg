import abc
import time
import traceback
from collections import OrderedDict
from typing import List, Tuple, Type, Dict, Any, Optional
from pathlib import Path

import torch
from torch import nn as nn

from rtg import TranslationExperiment as Experiment
from rtg import log, device, my_tensor as tensor, debug_mode
from rtg.dataprep import PAD_TOK, BOS_TOK, EOS_TOK, subsequent_mask
from rtg.binmt.bicycle import BiNMT
from rtg.module.rnnmt import RNNMT
from rtg.lm.rnnlm import RnnLm
from rtg.lm.tfmlm import TfmLm
from rtg.module.tfmnmt import TransformerNMT

Hypothesis = Tuple[float, List[int]]
StrHypothesis = Tuple[float, str]
INTERACTIVE = False


# TODO: simplify the generators
class GeneratorFactory(abc.ABC):

    def __init__(self, model, **kwargs):
        self.model = model

    @abc.abstractmethod
    def generate_next(self, past_ys):
        pass


class Seq2SeqGenerator(GeneratorFactory):

    def __init__(self, model: RNNMT, x_seqs, x_lens):
        super().__init__(model)
        # [S, B, d], [S, B, d] <-- [S, B], [B]
        self.enc_outs, enc_hids = model.encode(x_seqs, x_lens, None)
        # [S, B, d]
        self.dec_hids = enc_hids

    def generate_next(self, past_ys):
        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, _ = self.model.dec(self.enc_outs, last_ys, self.dec_hids)
        return log_probs


class BiNMTGenerator(Seq2SeqGenerator):

    def __init__(self, model: BiNMT, x_seqs, x_lens, path):
        # pick a sub Seq2Seq model inside the BiNMT model as per the given path
        assert path
        super().__init__(model.paths[path], x_seqs, x_lens)
        self.path = path
        self.wrapper = model


class T2TGenerator(GeneratorFactory):

    def __init__(self, model: TransformerNMT, x_seqs, x_lens=None):
        super().__init__(model)
        self.x_mask = (x_seqs != Decoder.pad_val).unsqueeze(1)
        self.memory = self.model.encode(x_seqs, self.x_mask)

    def generate_next(self, past_ys):
        out = self.model.decode(self.memory, self.x_mask, past_ys, subsequent_mask(past_ys.size(1)))
        log_probs = self.model.generator(out[:, -1])
        return log_probs


class ComboGenerator(GeneratorFactory):
    from rtg.syscomb import Combo

    def __init__(self, model: Combo, x_seqs, *args, **kwargs):
        super().__init__(model)
        self.x_mask = (x_seqs != Decoder.pad_val).unsqueeze(1)
        self.memory = self.model.encode(x_seqs, self.x_mask)

    def generate_next(self, past_ys):
        y_mask = subsequent_mask(past_ys.size(1))
        log_probs = self.model.generate_next(self.memory, self.x_mask, past_ys, y_mask)
        return log_probs


class RnnLmGenerator(GeneratorFactory):

    def __init__(self, model: RnnLm, x_seqs, x_lens):
        super().__init__(model)
        self.dec_hids = None
        if INTERACTIVE:
            # interactive mode use input as prefix
            n = x_seqs.shape[1] - 1 if x_seqs[0, -1] == EOS_TOK[1] else x_seqs.shape[1]
            for i in range(n):
                self.log_probs, self.dec_hids, _ = self.model(None, x_seqs[:, i], self.dec_hids)
            self.consumed = False

    def generate_next(self, past_ys):
        if INTERACTIVE and not self.consumed:
            assert past_ys[0, -1] == BOS_TOK[1]  # we are doing it right?
            self.consumed = True
            return self.log_probs

        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, _ = self.model(None, last_ys, self.dec_hids)
        return log_probs


class TfmLmGenerator(GeneratorFactory):

    def __init__(self, model: TfmLm, x_seqs, x_lens):
        super().__init__(model)
        if INTERACTIVE:
            self.x_seqs = x_seqs
            self.x_lens = x_lens
            for i in x_lens[1:]:
                # this feature was only meant to be used with a single sequence (probably beamed)
                # all seqs should've same length (else, padding assumption breaks in generate_next)
                assert x_lens[0] == i

    def generate_next(self, past_ys):
        if INTERACTIVE:
            # treat input (i.e. x_seqs) as a prefix for generation
            past_ys = torch.cat([self.x_seqs, past_ys], dim=1)

        seq_mask = subsequent_mask(past_ys.size(1))
        out = self.model(past_ys, seq_mask, gen_probs=False)
        # only generate probs for the last time step
        log_probs = self.model.generator(out[:, -1])
        return log_probs


generators = {'t2t': T2TGenerator,
              'seq2seq': Seq2SeqGenerator,
              'binmt': BiNMTGenerator,
              'combo': ComboGenerator,
              'tfmnmt': T2TGenerator,
              'rnnmt': Seq2SeqGenerator,
              'rnnlm': RnnLmGenerator,
              'tfmlm': TfmLmGenerator
              }
factories = {
    't2t': TransformerNMT.make_model,
    'seq2seq': RNNMT.make_model,
    'binmt': BiNMT.make_model,
    'tfmnmt': TransformerNMT.make_model,
    'rnnmt': RNNMT.make_model,
    'rnnlm': RnnLm.make_model,
    'tfmlm': TfmLm.make_model
}


def load_models(models: List[Path], exp: Experiment):
    res = []
    for i, model_path in enumerate(models):
        assert model_path.exists()
        log.info(f"Load Model {i}: {model_path} ")
        chkpt = torch.load(str(model_path), map_location=device)
        model = instantiate_model(chkpt)
        res.append(model)
    return res


def instantiate_model(checkpt_state, exp=None):
    chkpt = checkpt_state
    state = chkpt['model_state']
    model_type = chkpt['model_type']
    model_args = chkpt['model_args']
    # Dummy experiment wrapper
    factory = factories[model_type]
    model = factory(exp=exp, **model_args)[0]
    model.load_state_dict(state)
    log.info(f"Successfully restored the model state of : {model_type}")
    return model


class ReloadEvent(Exception):
    """An exception to reload model with new path
    -- Its a kind of hack to pass event back to caller and redo interactive shell--
    """

    def __init__(self, model_paths, state: Dict[str, Any]):
        super().__init__()
        self.model_paths = model_paths
        self.state = state


class Decoder:
    pad_val = PAD_TOK[1]
    bos_val = BOS_TOK[1]
    eos_val = EOS_TOK[1]
    default_beam_size = 5

    def __init__(self, model, gen_factory: Type[GeneratorFactory], exp, gen_args=None,
                 debug=debug_mode):
        self.model = model
        self.exp = exp
        self.gen_factory = gen_factory
        self.debug = debug
        self.gen_args = gen_args if gen_args is not None else {}

    def generator(self, x_seqs, x_lens):
        return self.gen_factory(self.model, x_seqs=x_seqs, x_lens=x_lens, **self.gen_args)

    @staticmethod
    def average_states(state_dict: OrderedDict, *state_dicts: OrderedDict):
        w = 1.0 / (1 + len(state_dicts))
        if state_dicts:
            key_set = set(state_dict.keys())
            assert all(key_set == set(st.keys()) for st in state_dicts)
            for key in key_set:
                state_dict[key] *= w
                for st in state_dicts:
                    state_dict[key] += w * st[key]
        return state_dict

    @staticmethod
    def _checkpt_to_model_state(checkpt_path: str):
        state = torch.load(checkpt_path, map_location=device)
        if 'model_state' in state:
            state = state['model_state']
        return state

    @staticmethod
    def maybe_ensemble_state(exp, model_paths: Optional[List[str]], ensemble: int = 1):
        if model_paths and len(model_paths) == 1:
            log.info(f" Restoring state from requested model {model_paths[0]}")
            return Decoder._checkpt_to_model_state(model_paths[0])
        elif not model_paths and ensemble <= 1:
            model_path, _ = exp.get_best_known_model()
            log.info(f" Restoring state from best known model: {model_path}")
            return Decoder._checkpt_to_model_state(model_path)
        else:
            if not model_paths:
                # Average
                model_paths = exp.list_models()[:ensemble]
            log.info(f"Averaging {len(model_paths)} model states :: {model_paths}")
            states = [Decoder._checkpt_to_model_state(mp) for mp in model_paths]
            return Decoder.average_states(*states)

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
            state = cls.maybe_ensemble_state(exp, model_paths=model_paths, ensemble=ensemble)
            model.load_state_dict(state)
            log.info("Successfully restored the model state.")
        elif isinstance(model, nn.DataParallel):
            model = model.module

        model = model.eval().to(device=device)
        generator = generators[model_type]
        return cls(model, generator, exp, gen_args)

    def greedy_decode(self, x_seqs, x_lens, max_len, **args) -> List[Hypothesis]:
        """
        Implements a simple greedy decoder
        :param x_seqs:
        :param x_lens: length of x sequences
        :param max_len:
        :return:
        """

        gen = self.generator(x_seqs, x_lens)
        batch_size = x_seqs.size(0)
        ys = torch.full(size=(batch_size, 1), fill_value=self.bos_val, dtype=torch.long,
                        device=device)
        scores = torch.zeros(batch_size, device=device)

        actives = ys[:, -1] != self.eos_val
        for i in range(1, max_len + 1):
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

    def beam_decode(self, x_seqs, x_lens, max_len, beam_size=default_beam_size, num_hyp=None,
                    **args) -> List[List[Hypothesis]]:
        """
        :param x_seqs: input batch of sequences
        :param max_len:  maximum length to consider if decoder doesnt produce EOS token
        :param beam_size: beam size
        :param num_hyp: number of hypothesis in each beam to return
        :param args:
        :return: List of num_hyp Hypothesis for each sequence in the batch.
         Each hypothesis consists of score and a list of word indexes.
        """
        # TODO: rewrite this, this function is a mess!
        # repeat beam size
        batch_size = x_seqs.size(0)
        assert batch_size == 1  # TODO: test large batches
        if not num_hyp:
            num_hyp = beam_size
        beam_size = max(beam_size, num_hyp)

        # Everything beamed_*  below is the batch repeated beam_size times
        beamed_batch_size = batch_size * beam_size

        beamed_x_seqs = x_seqs.repeat(1, beam_size).view(beamed_batch_size, -1)
        beamed_x_lens = x_lens.view(-1, 1).repeat(1, beam_size).view(beamed_batch_size)
        generator = self.generator(beamed_x_seqs, beamed_x_lens)

        beamed_ys = torch.full(size=(beamed_batch_size, 1), fill_value=self.bos_val,
                               dtype=torch.long, device=device)
        beamed_scores = torch.zeros((beamed_batch_size, 1), device=device)

        beam_active = torch.ones((beamed_batch_size, 1), dtype=torch.uint8, device=device)
        # zeros means ended, one means active
        for t in range(1, max_len + 1):
            if beam_active.sum() == 0:
                break
            # [batch*beam, Vocab]
            log_prob = generator.generate_next(beamed_ys)

            # broad cast scores along row (broadcast) and sum  log probabilities
            next_scores = beamed_scores + beam_active.float() * log_prob  # Zero out inactive beams

            top_scores, nxt_idx = next_scores.topk(k=beam_size,
                                                   dim=-1)  # [batch*beam, beam],[batch*beam, beam]
            # Now we got beam_size*beam_size heads, task: shrink it to beam_size
            # Since the ys will change, after re-scoring, we will make a new tensor for new ys
            new_ys = torch.full(size=(beamed_batch_size, beamed_ys.size(1) + 1),
                                fill_value=self.pad_val,
                                device=device, dtype=torch.long)

            for i in range(batch_size):
                # going to picking top k out of k*k beams for each sequence in batch
                # beams of i'th sequence in batch have this start and end
                start, end = i * beam_size, (i + 1) * beam_size
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
                    seqi_nxt_ys = torch.full((beam_size, 1), fill_value=self.pad_val, device=device)
                    seqi_top_scores = torch.zeros((beam_size, 1), device=device)

                    # ignore the inactive beams, don't grow them any further
                    # INACTIVE BEAMS: Preserve the inactive beams, just copy them
                    seqi_inactive_mask = (beam_active[start:end, -1] == 0).view(-1, 1)
                    seqi_inactive_count = seqi_inactive_mask.sum()
                    active_start = start + seqi_inactive_count  # [start, ... active_start-1, active_start, ... end]
                    if seqi_inactive_count > 0:  # if there are some inactive beams
                        seqi_inactive_ys = self.masked_select(beamed_ys[start:end, :],
                                                              seqi_inactive_mask)
                        new_ys[start: active_start, :-1] = seqi_inactive_ys  # Copy inactive beams
                        seqi_top_scores[start:active_start, :] = \
                            self.masked_select(beamed_scores[start:end, :], seqi_inactive_mask)

                    # ACTIVE BEAMS: the remaining beams should be let to grow
                    seqi_active_mask = (beam_active[start:end, -1] == 1).view(-1, 1)
                    seqi_active_count = seqi_active_mask.sum()  # task is to select these many top scoring beams

                    seqi_scores = self.masked_select(top_scores[start:end, :], seqi_active_mask)
                    seqi_nxt_idx = self.masked_select(nxt_idx[start:end, :], seqi_active_mask)

                    seqi_active_top_scores, seqi_nxt_idx_idx = seqi_scores.view(-1).topk(
                        k=seqi_active_count)
                    seqi_top_scores[active_start:end, 0] = seqi_active_top_scores
                    seqi_nxt_ys[active_start: end, 0] = seqi_nxt_idx.view(-1)[seqi_nxt_idx_idx]

                    # Select active ys
                    active_beam_idx = seqi_nxt_idx_idx / beam_size
                    seqi_active_ys = self.masked_select(beamed_ys[start:end, :], seqi_active_mask)
                    seqi_active_old_ys = seqi_active_ys.index_select(0, active_beam_idx)
                    new_ys[active_start:end, :-1] = seqi_active_old_ys

                    # Update status of beam_active flags
                    beam_active[active_start:end, :] = self.masked_select(beam_active[start:end, :],
                                                                          seqi_active_mask) \
                        .index_select(0, active_beam_idx)
                    if active_start > start:
                        beam_active[start:active_start, -1] = 0  # inactive beams are set to zero

                beamed_scores[start:end, :] = seqi_top_scores
                # copy the new word indices to last step
                new_ys[start:end, -1] = seqi_nxt_ys.view(-1)
            beamed_ys = new_ys
            # AND to update active flag
            beam_active = beam_active & (beamed_ys[:, -1] != self.eos_val).view(beamed_batch_size,
                                                                                1)

        result = []
        # reverse sort based on the score
        for i in range(batch_size):
            result.append([])
            start, end = i * beam_size, (i + 1) * beam_size
            scores, indices = beamed_scores[start:end, :].view(-1).sort(descending=True)
            for j in range(beam_size):
                if len(result[-1]) == num_hyp:
                    continue
                result[-1].append(
                    (scores[j].item(), beamed_ys[start + indices[j], 1:].squeeze().tolist()))
        return result

    @property
    def inp_vocab(self):
        # the choice of vocabulary can be tricky, because of bidirectional model
        if self.exp.model_type == 'binmt':
            return {
                'E1': self.exp.src_vocab,
                'E2': self.exp.tgt_vocab
            }[self.gen_args['path'][:2]]
        else:  # all others go from source as input to target as output
            return self.exp.src_vocab

    @property
    def out_vocab(self):
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
            in_seq = self.inp_vocab.encode_as_ids(line, add_eos=True, add_bos=True)
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

    # noinspection PyUnresolvedReferences
    def decode_interactive(self, **args):
        import sys
        import readline
        helps = [(':quit', 'Exit'),
                 (':help', 'Print this help message'),
                 (':beam_size <n>', 'Set beam size to n'),
                 (':num_hyp <k>', 'Print top k hypotheses'),
                 (':debug', 'Enable debug mode'),
                 (':-debug', 'Disable debug mode'),
                 (':models', 'show all available models of this experiment'),
                 (':model <number>', 'reload shell with the model chosen by <number>')
                 ]
        if self.exp.model_type == 'binmt':
            helps.append((':path <path>', 'BiNMT modules: {E1D1, E2D2, E1D2E2D1, E2D2E1D2}'))

        def print_cmds():
            for cmd, msg in helps:
                print(f"\t{cmd:15}\t-\t{msg}")

        print("Launching Interactive shell...")
        global INTERACTIVE
        INTERACTIVE = True
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
                elif line.startswith(":beam_size "):
                    args['beam_size'] = int(line.replace(':beam_size', '').strip())
                    print_state = True
                elif line.startswith(":num_hyp"):
                    args['num_hyp'] = int(line.replace(':num_hyp', '').strip())
                    print_state = True
                elif line.startswith(":debug"):
                    self.debug = True
                    args['debug'] = True
                    print_state = True
                elif line.startswith(":-debug"):
                    self.debug = False
                    print_state = True
                elif line.startswith(":path"):
                    self.gen_args['path'] = line.replace(':path', '').strip()
                    print_state = True
                elif line.startswith(":models"):
                    for i, mod_path in enumerate(self.exp.list_models()):
                        print(f"\t{i}\t{mod_path}")
                elif line.startswith(":model"):
                    mod_idxs = [int(x) for x in line.replace(":model", "").strip().split()]
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
                    print(f'\t|took={1000 * (time.time()-start):.3f}ms')
                    for score, hyp in res:
                        print(f'  {score:.4f}\t{hyp}')
            except ReloadEvent as re:
                raise re  # send it to caller
            except EOFError as e1:
                break
            except Exception:
                traceback.print_exc()
                print_state = True

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
