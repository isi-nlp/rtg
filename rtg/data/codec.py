#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 4/18/20

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Iterator, Union, Optional
import collections as coll
from tqdm import tqdm
import numpy as np
from rtg import log, yaml
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from rtg.utils import IO

Array = np.ndarray


class Field(metaclass=ABCMeta):
    pad_tok, pad_idx = '<pad>', 0
    unk_tok, unk_idx = '<unk>', 1
    bos_tok, bos_idx = '<s>', 2
    eos_tok, eos_idx = '</s>', 3
    cls_tok, cls_idx = '<cls>', 4
    reserved_toks = [pad_tok, unk_tok, bos_tok, eos_tok, cls_tok]
    reserved_idxs = [pad_idx, unk_idx, bos_idx, eos_idx, cls_idx]

    def __init__(self):
        self.class_names = None

    @classmethod
    def reserved(cls):
        return list(zip(cls.reserved_toks, cls.reserved_idxs))

    @abstractmethod
    def encode_as_ids(self, text, add_bos, add_eos, split_ratio: Optional[float] = 0.) -> Array:
        raise NotImplementedError

    @abstractmethod
    def decode_ids(self, ids, trunc_eos):
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text):
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, tokens):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train(model_type: str, vocab_size: int, model_path: str, files: Iterator[str],
              no_split_toks: Optional[List[str]] = None,
              char_coverage: float = -1.0):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :param no_split_toks: Don't split these tokens
        :return:
        """
        raise NotImplementedError

    def shrink_vocab(self, files: List, min_freq:int, save_at: Path) -> List[int]:
        """
        Shrinks the current vocabulary and saves at given path
        :param files: corpus file to extract vocab
        :param min_freq: minimum frequency that determines whether a type remains or drops out
        :param save_at: path to save the modified vocab
        :return:  List of indexes. Example [0, 2, 5] is spec for shrinking 6 items to 3.
         Items at index 0 remains at 0; index 2 goes to index 1, and 5 goes to index 2.
         Indices are sorted as increasing order, so its safe to do inplace updates from index 0.
        """
        raise Exception(f'Not implemented for {type(self)}')


class SPField(SentencePieceProcessor, Field):
    """A wrapper class for sentence piece trainer and processor"""

    # mask_tok, mask_idx = '<mask>', 5   # TODO: support <mask>

    def __init__(self, path: str):
        super().__init__()
        assert self.load(path)
        self.class_names = [self.IdToPiece(i) for i in range(len(self))]

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False, split_ratio=0.) -> Array:
        assert split_ratio == 0, 'SentencePiece doesnt support SWR, ' \
                                 'please use NLCodec or disable SWR by setting split_ratio=0'

        ids = super(SPField, self).encode_as_ids(text)
        if add_bos and ids[0] != self.bos_idx:
            ids.insert(0, self.bos_idx)
        if add_eos and ids[-1] != self.eos_idx:
            ids.append(self.eos_idx)
        return np.array(ids, dtype=np.int32)

    def decode_ids(self, ids: List[int], trunc_eos=False) -> str:
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        if trunc_eos:
            try:
                ids = ids[:ids.index(self.eos_idx)]
            except ValueError:
                pass
        return super(SPField, self).decode_ids(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.encode_as_pieces(text.encode())

    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens).replace('▁', ' ').strip()

    @classmethod
    def train(cls, model_type: str, vocab_size: int, model_path: str, files: Iterator[str],
              no_split_toks: Optional[List[str]] = None, char_coverage: float = 0):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :param no_split_toks: Don't split these tokens
        :param char_coverage: character coverage (0, 1]. value <= 0 => default coverage 0.9995%
        :return:
        """
        model_prefix = model_path.replace('.model', '')
        files = ','.join(files)  # remove duplicates
        arg = f"--input={files} --vocab_size={vocab_size} --model_prefix={model_prefix}" \
              f" --model_type={model_type} --pad_id={cls.pad_idx} --bos_id={cls.bos_idx}" \
              f" --eos_id={cls.eos_idx} --unk_id={cls.unk_idx} --hard_vocab_limit=false"
        if char_coverage > 0:
            assert 0 < char_coverage <= 1
            arg += f" --character_coverage={char_coverage}"
        # CLS token goes in the beginning because we need it get index 4
        extra = [cls.cls_tok] + (no_split_toks or [])
        no_split_toks_str = ','.join(extra)
        arg += f" --user_defined_symbols={no_split_toks_str}"
        if model_type == 'bpe':  # BPE can have longer sentences, default is 2048
            arg += " --max_sentence_length=8192"
        if model_type == 'word':
            arg += ' --use_all_vocab'
        log.info(f"SPM: {arg}")
        SentencePieceTrainer.Train(arg)
        log.info("Training complete")
        if not model_path.endswith('.model'):
            model_path += '.model'
        model = SPField(model_path)
        for piece, idx in cls.reserved():
            assert model.piece_to_id(piece) == idx
        return model


class NLField(Field):
    # from nlcodec lib

    def __init__(self, path: Union[str, Path]):
        # this is experimental
        from nlcodec import load_scheme, EncoderScheme, Type
        self.codec: EncoderScheme = load_scheme(path)
        self.vocab: List[Type] = self.codec.table
        log.info(f'Loaded {len(self.codec)} types from {path}')
        if self.codec.name not in ('class',):  # except in classification field
            for tok, idx in self.reserved():  # reserved are reserved
                # Todo swap it with nlcodec.Reserved
                assert self.vocab[idx].name == tok
        self.class_names = [t.name for t in self.vocab]

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False, split_ratio=0.) -> Array:
        if self.codec.name == "bpe" and split_ratio > 0:
            ids = self.codec.encode(text, split_ratio)
        else:
            ids = self.codec.encode(text)

        if add_bos and ids[0] != self.bos_idx:
            ids.insert(0, self.bos_idx)
        if add_eos and ids[-1] != self.eos_idx:
            ids.append(self.eos_idx)
        return np.array(ids, dtype=np.int32)

    def decode_ids(self, ids: List[int], trunc_eos=False, remove_pads=True) -> str:
        if trunc_eos:
            try:
                ids = ids[:ids.index(self.eos_idx)]
            except ValueError:
                pass
        if remove_pads:
            ids = [i for i in ids if i != self.pad_idx]
        return self.codec.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.codec.encode_str(text)

    def detokenize(self, tokens: List[str]) -> str:
        return self.codec.decode_str(tokens)

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def train(cls, model_type: str, vocab_size: int, model_path: str, files: List[str],
              no_split_toks: Optional[List[str]] = None, char_coverage: float = 0,
              dedup=True, spark=None, min_co_ev=None):
        """
        :param model_type: word, char, bpe
        :param vocab_size: vocabulary size
        :param model_path: where to store vocabulary model
        :param files: text for creating vcabulary
        :param no_split_toks:
        :param char_coverage: character coverage (0, 1]. value <= 0 => default coverage
        :param min_co_ev: (for BPE only) minimum co-evidence for subword merges
        :return:
        """
        assert not no_split_toks, 'not supported in nlcodec yet'
        from nlcodec import learn_vocab, term_freq
        kwargs = dict(char_coverage=char_coverage) if char_coverage > 0 else {}
        if min_co_ev:
            kwargs["min_co_ev"] = min_co_ev
        if not spark:
            inp = IO.get_liness(*files)
        else:
            # extract and store frequencies to this file
            stats_file = model_path  + '.termfreqs'
            if not Path(stats_file).exists():
                log.info("Extracting term frequencies... ")
                paths = [f if isinstance(f, Path) else Path(f) for f in files]
                wfs, chfs, n_lines = term_freq.word_counts(paths=paths, dedup=dedup, spark=spark)
                log.info(f"Lines = {n_lines:,}, Word Types: {len(wfs):,} Char Types:{len(chfs):,}")
                stats = chfs if model_type == 'char' else wfs
                log.info(f"Writing frequencies to {stats_file}")
                with IO.writer(stats_file) as out:
                    term_freq.write_stats(stats=stats, out=out, line_count=n_lines)
                kwargs['term_freqs'] = True
            inp = IO.get_lines(stats_file, delim='\n')

        learn_vocab(inp=inp, level=model_type, model=model_path, vocab_size=vocab_size, **kwargs)
        return cls(model_path)

    def shrink_vocab(self, files: List, min_freq:int, save_at: Path) -> List[int]:
        """
        Shrinks the current vocabulary and saves at given path
        :param files: corpus file to extract vocab
        :param min_freq: minimum frequency that determines whether a type remains or drops out
        :param save_at: path to save the modified vocab
        :return:  List of indexes. Example [0, 2, 5] is spec for shrinking 6 items to 3.
         Items at index 0 remains at 0; index 2 goes to index 1, and 5 goes to index 2.
         Indices are sorted as increasing order, so its safe to do inplace updates from index 0.
        """
        mappings = self.codec.shrink_vocab(files, min_freq=min_freq, save_at=save_at)
        return mappings

class PretrainMatchField(Field):
    # this order is for fairseq's XML-R

    """
    bos_tok, bos_idx = '<s>', 0
    pad_tok, pad_idx = '<pad>', 1
    eos_tok, eos_idx = '</s>', 2
    unk_tok, unk_idx = '<unk>', 3
    reserved_idxs = [pad_idx, unk_idx, bos_idx, eos_idx]
    reserved_toks = [pad_tok, unk_tok, bos_tok, eos_tok]
    """

    def __init__(self, path: Union[str, Path]):
        with IO.reader(path) as rdr:
            data = yaml.load(rdr)
        hub_api = self.load_hub_model(data['model_id'])
        # these are for XML-R wiz RoBERTa from fairseq  ; generalize it for other models later
        self.bpe = hub_api.bpe

        self.tok2idx = {tok:new_idx for tok, (new_idx, old_idx) in data['mapping'].items()}
        self.idx2tok = list(sorted(self.tok2idx.keys(), key=self.tok2idx.get, reverse=False))
        assert len(self.idx2tok) == len(self.tok2idx)

        for tok, idx in self.reserved():  # reserved are reserved
            assert self.tok2idx[tok] == idx
            assert self.idx2tok[idx] == tok
        self.new_idx2old_idx = {new_idx: old_idx for tok, (new_idx, old_idx) in data['mapping'].items()}
        self.class_names = self.idx2tok

    @classmethod
    def load_hub_model(cls, model_id):
        github, model_name = model_id.split(':')
        from torch.hub import load as load_model
        hub_api = load_model(github, model_name)
        return hub_api

    @classmethod
    def train(cls, model_type: str, vocab_size: int, model_path: Union[Path, str], files: List[str],
              tok_coverage=0.9999, **kwargs):
        # Note: char_coverage is abused as subword_coverage
        hub_api = cls.load_hub_model(model_type)
        bpe = hub_api.bpe
        dicto = hub_api.task.dictionary

        freqs = coll.Counter()
        lines = IO.get_liness(*files)
        for line in tqdm(lines, mininterval=2, dynamic_ncols=True, unit='line'):
            freqs.update(bpe.encode(line).split())
        total_toks = sum(freqs.values())
        log.info(f"Found {len(freqs)} bpe types and {total_toks} toks")

        freqs = list(sorted(freqs.items(), reverse=True, key=lambda x: x[1]))
        vocabulary, oovs = [], []
        cumulative = 0
        for t, f in freqs:
            if cumulative / total_toks <= tok_coverage:
                vocabulary.append((t, f))
                cumulative += f
            else:
                oovs.append((t, f))

        oovs_str = ' '.join(f'{t}:{f}' for t, f in oovs)
        log.info(f'Excluded {len(oovs)} types as OOVs.\n:{oovs_str}')
        log.info(f'Included {len(vocabulary)} types as in vocabulary; '
                    f'Coverage = {cumulative / total_toks:g}')
        # TODO: mapping should be list[int] with one on one map
        types, indices = [], {}
        for typ, new_idx in cls.reserved():
            assert len(types) == new_idx
            types.append(typ)
            old_idx = dicto.indices.get(typ, -1)
            indices[typ] = [new_idx, old_idx]

        for typ, freq in vocabulary:
            # [new index, old index]
            indices[typ] = [len(types), dicto.indices.get(typ, -1)]
            types.append(typ)

        data = {
            'model_id': model_type,
            'mapping': indices
        }
        with IO.writer(model_path) as wrtr:
            yaml.dump(data, wrtr)
        return cls(model_path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False, split_ratio=0.) -> Array:
        assert split_ratio == 0, 'SentencePiece doesnt support SWR, ' \
                                 'please use NLCodec or disable SWR by setting split_ratio=0'

        pieces = self.tokenize(text)
        ids = [self.tok2idx.get(p, self.unk_idx) for p in pieces]
        if add_bos and ids[0] != self.bos_idx:
            ids.insert(0, self.bos_idx)
        if add_eos and ids[-1] != self.eos_idx:
            ids.append(self.eos_idx)
        return np.array(ids, dtype=np.int32)

    def decode_ids(self, ids: List[int], trunc_eos=False, remove_pads=True) -> str:
        if trunc_eos:
            try:
                ids = ids[:ids.index(self.eos_idx)]
            except ValueError:
                pass
        if remove_pads:
            ids = [i for i in ids if i != self.pad_idx]
        pieces = [self.idx2tok[i] for i in ids]
        return self.detokenize(pieces)

    def tokenize(self, text: str) -> List[str]:
        return self.bpe.encode(text).split()

    def detokenize(self, tokens: List[str]) -> str:
        return self.bpe.decode(' '.join(tokens))

    def __len__(self):
        return len(self.idx2tok)
