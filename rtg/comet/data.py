from collections import namedtuple
from typing import Callable, List, Any
from dataclasses import dataclass

import numpy as np
import torch

from rtg import Array, device, dtorch, get_my_args, log, register_model
from rtg.data.codec import Field as BaseField

Example = namedtuple('IdExample', ['id', 'x1', 'x2', 'y'])

@dataclass(frozen=True)
class Example:
    
    id: Any
    x1: Array
    x2: Array
    y: Array
        
    @classmethod
    def new_with_length_check(cls, id, x1, x2, y, max_src_len:int, max_tgt_len:int):
        return cls(id, x1[:max_src_len], x2[:max_src_len], y[:max_tgt_len])


class Batch:
    def __init__(self, buffer: List[Example], fields, device=device) -> None:
        batch_size = len(buffer)
        assert len(fields) == 3

        max_lens = dict(x1=0, x2=0, y=0)
        for ex in buffer:
            max_lens['x1'] = max(max_lens['x1'], len(ex.x1))
            max_lens['x2'] = max(max_lens['x2'], len(ex.x2))
            max_lens['y'] = max(max_lens['y'], len(ex.y))

        assert fields[0].pad_idx == fields[1].pad_idx
        self.pad_val = fields[0].pad_idx
        self.x1s = torch.full(
            (batch_size, max_lens['x1']), fill_value=self.pad_val, dtype=torch.long, device=device
        )
        self.x2s = torch.full(
            (batch_size, max_lens['x2']), fill_value=self.pad_val, dtype=torch.long, device=device
        )
        # y is class. it doesnt require padding
        self.ys = torch.zeros((batch_size, max_lens['y']), dtype=torch.long, device=device)
        for idx, ex in enumerate(buffer):
            self.x1s[idx, : len(ex.x1)] = torch.tensor(ex.x1, dtype=torch.long, device=device)
            self.x2s[idx, : len(ex.x2)] = torch.tensor(ex.x2, dtype=torch.long, device=device)
            self.ys[idx, : len(ex.y)] = torch.tensor(ex.y, dtype=torch.long, device=device)
        # [B, 1] -> [B] for classification
        self.ys = self.ys.squeeze(1)

    def to(self, device):
        self.x1s = self.x1s.to(device)
        self.x2s = self.x2s.to(device)
        self.ys = self.ys.to(device)
        return self

    def __len__(self):
        return len(self.x1s)


class HFField(BaseField):
    def __init__(self, model_id):
        super().__init__()

        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.class_names = self.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))
        # link the special token ids/idxs
        self.bos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.pad_idx = self.tokenizer.pad_token_id
        self.unk_idx = self.tokenizer.unk_token_id
        self.cls_idx = self.tokenizer.cls_token_id
        self.sep_idx = self.tokenizer.sep_token_id
        self.mask_idx = self.tokenizer.mask_token_id

    def __len__(self):
        return self.tokenizer.vocab_size

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False, split_ratio=0.0) -> Array:
        assert split_ratio == 0, (
            'SentencePiece doesnt support SWR, ' 'please use NLCodec or disable SWR by setting split_ratio=0'
        )
        ids = self.tokenizer.encode(text)
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
                ids = ids[: ids.index(self.eos_idx)]
            except ValueError:
                pass
        return self.tokenizer.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens)

    @classmethod
    def train(cls, *args, **kwargs):
        raise Exception('Training is not supported for HFField')
