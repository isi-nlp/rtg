#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/18/20

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Iterator, Union, Optional

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from rtg import log, yaml
from rtg.utils import IO


class Field(metaclass=ABCMeta):

    pad_tok, pad_idx = '<pad>', 0
    unk_tok, unk_idx = '<unk>', 1
    bos_tok, bos_idx = '<s>', 2
    eos_tok, eos_idx = '</s>', 3
    cls_tok, cls_idx = '<cls>', 4
    reserved_toks = [pad_tok, unk_tok, bos_tok, eos_tok, cls_tok]
    reserved_idxs = [pad_idx, unk_idx, bos_idx, eos_idx, cls_idx]

    @classmethod
    def reserved(cls):
        return list(zip(cls.reserved_toks, cls.reserved_idxs))

    @abstractmethod
    def encode_as_ids(self, text, add_bos, add_eos):
        pass

    @abstractmethod
    def decode_ids(self, ids, trunc_eos):
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def detokenize(self, tokens):
        pass

    @abstractmethod
    def __len__(self):
        pass

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
        pass


class SPField(SentencePieceProcessor, Field):
    """A wrapper class for sentence piece trainer and processor"""
    #mask_tok, mask_idx = '<mask>', 5   # TODO: support <mask>

    def __init__(self, path: str):
        super().__init__()
        assert self.load(path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = super(SPField, self).encode_as_ids(text)
        if add_bos and ids[0] != self.bos_idx:
            ids.insert(0, self.bos_idx)
        if add_eos and ids[-1] != self.eos_idx:
            ids.append(self.eos_idx)
        return ids

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
        for tok, idx in self.reserved():  # reserved are reserved
            # Todo swap it with nlcodec.Reserved
            assert self.vocab[idx].name == tok

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = self.codec.encode(text)
        if add_bos and ids[0] != self.bos_idx:
            ids.insert(0, self.bos_idx)
        if add_eos and ids[-1] != self.eos_idx:
            ids.append(self.eos_idx)
        return ids

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
              no_split_toks: Optional[List[str]] = None, char_coverage: float = 0):
        """

        :param model_type: word, char, bpe
        :param vocab_size: vocabulary size
        :param model_path: where to store vocabulary model
        :param files: text for creating vcabulary
        :param no_split_toks:
        :param char_coverage: character coverage (0, 1]. value <= 0 => default coverage
        :return:
        """
        from nlcodec import learn_vocab
        inp = IO.get_liness(*files)
        assert not no_split_toks, 'not supported in nlcodec yet'
        kwargs = dict(char_coverage=char_coverage) if char_coverage > 0 else {}
        learn_vocab(inp=inp, level=model_type, model=model_path, vocab_size=vocab_size, **kwargs)
        return cls(model_path)


class PretrainMatchField(Field):

    # this order is for fairseq's XML-R

    bos_tok, bos_idx = '<s>', 0
    pad_tok, pad_idx = '<pad>', 1
    eos_tok, eos_idx = '</s>', 2
    unk_tok, unk_idx = '<unk>', 3
    reserved_idxs = [pad_idx, unk_idx, bos_idx, eos_idx]
    reserved_toks = [pad_tok, unk_tok, bos_tok, eos_tok]

    def __init__(self, path: Union[str, Path]):
        with IO.reader(path) as rdr:
            data = yaml.load(rdr)
        github, model_name = data['model_id'].split(':')
        from torch.hub import load as load_model
        hub_api = load_model(github, model_name)
        # these are for XML-R wiz RoBERTa from fairseq  ; generalize it for other models later
        self.bpe = hub_api.bpe
        self.dicto = hub_api.task.dictionary

        for tok, idx in self.reserved():  # reserved are reserved
            assert self.dicto.indices[tok] == idx
            assert self.dicto.symbols[idx] == tok

    @classmethod
    def train(cls, model_type: str, vocab_size: int, model_path: Union[Path, str], *args, **kwargs):
        # IDEA: shrink vocabulary to only include training data types
        data = {'model_id': model_type}
        with IO.writer(model_path) as wrtr:
            yaml.dump(data, wrtr)
        return cls(model_path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        pieces = self.tokenize(text)
        ids = [self.dicto.indices.get(p, self.unk_idx) for p in pieces]
        if add_bos and ids[0] != self.bos_idx:
            ids.insert(0, self.bos_idx)
        if add_eos and ids[-1] != self.eos_idx:
            ids.append(self.eos_idx)
        return ids

    def decode_ids(self, ids: List[int], trunc_eos=False, remove_pads=True) -> str:
        if trunc_eos:
            try:
                ids = ids[:ids.index(self.eos_idx)]
            except ValueError:
                pass
        if remove_pads:
            ids = [i for i in ids if i != self.pad_idx]
        pieces = [self.dicto.symbols[i] for i in ids]
        return self.detokenize(pieces)

    def tokenize(self, text: str) -> List[str]:
        return self.bpe.encode(text).split()

    def detokenize(self, tokens: List[str]) -> str:
        return self.bpe.decode(' '.join(tokens))

    def __len__(self):
        return len(self.dicto)
