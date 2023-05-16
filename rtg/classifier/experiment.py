#!/usr/bin/env python


from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.cuda.amp import autocast

from rtg import ProblemType, device, log, BatchIterable
from rtg.registry import MODEL, ProblemType
from rtg.eval.clsmetric import ClsMetric
from rtg.utils import IO
from rtg.nmt import TranslationExperiment


class ClassificationExperiment(TranslationExperiment):
    """
    Treat source as source sequence, target as class
    translation is many:many, classification is many:1, a special case of many:many
    """

    def __init__(self, *args, **kwargs):
        super(ClassificationExperiment, self).__init__(*args, **kwargs)
        self.train_db = self.data_dir / 'train.nldb'

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.CLASSIFICATION

    @property
    def src_to_ids(self):
        # EOS is added by batch maker during training
        return partial(self.src_field.encode_as_ids, add_bos=False, add_eos=True)

    def get_val_data(
        self,
        batch_size: Union[int, Tuple[int, int]],
        sort_desc=False,
        batch_first=True,
        shuffle=False,
        y_is_cls=False,
    ):
        return BatchIterable(
            self.valid_file,
            batch_size=batch_size,
            sort_desc=sort_desc,
            batch_first=batch_first,
            shuffle=shuffle,
            field=self.tgt_vocab,
            keep_in_mem=True,
            y_is_cls=y_is_cls,
            **self._get_batch_args(),
        )

    def pre_process(self, args=None, force=False):
        args = args or self.config.get('prep')
        is_shared = args.get('shared')
        assert not is_shared, 'Shared vocab not supported for Classification.' ' Please set prep.shared=False'
        # skip TranslationExperiment, go to its parent BaseExperiment pre_process
        super(TranslationExperiment, self).pre_process(args=args, force=force)

        if self.has_prepared() and not force:
            log.warning("Already prepared")
            return

        if 'parent' in self.config:
            self.inherit_parent()

        # check if files are parallel
        self.check_line_count('validation', args['valid_src'], args['valid_tgt'])
        xt_args = dict(
            no_split_toks=args.get('no_split_toks'),
            char_coverage=args.get('char_coverage', 0),
            min_co_ev=args.get('src_min_co_ev', args.get('min_co_ev', None)),
        )
        if force or not self._src_field_file.exists():
            src_corpus = []
            if args.get('train_src') and not args.get('train_src').startswith('stdin:'):
                src_corpus.append(args['train_src'])
            if args.get('mono_src'):
                src_corpus.append(args['mono_src'])

            assert src_corpus, 'prep.train_src (not stdin) or prep.mono_src must be defined'
            max_src_size = args.get('max_src_types', args.get('max_types', None))
            assert max_src_size, 'prep.max_src_types or prep.max_types must be defined'
            self.src_field = self._make_vocab(
                "src",
                self._src_field_file,
                args['pieces'],
                vocab_size=max_src_size,
                corpus=src_corpus,
                **xt_args,
            )

        if force or not self._tgt_field_file.exists():
            # target vocabulary; class names. treat each line as a word
            tgt_corpus = []
            if args.get('train_tgt') and not args.get('train_tgt').startswith('stdin:'):
                tgt_corpus.append(args['train_tgt'])
            if args.get('mono_tgt'):
                tgt_corpus.append(args['mono_tgt'])
            assert tgt_corpus, 'prep.train_tgt (not stdin) or prep.mono_tgt must be defined'

            self.tgt_field = self._make_vocab(
                "tgt", self._tgt_field_file, 'class', corpus=tgt_corpus, vocab_size=-1
            )
        n_classes = self.config['model_args'].get('tgt_vocab')
        if len(self.tgt_field) != n_classes:
            log.warning(
                f'model_args.tgt_vocab={n_classes},' f' but found {len(self.tgt_field)} cls in {tgt_corpus}'
            )

        train_file = self.train_db
        if args.get('train_src', '').startswith('stdin:') or args.get('train_tgt', '').startswith('stdin:'):
            log.info('skipping binarization of training data since it is stdin')
        else:
            self._pre_process_parallel(
                'train_src', 'train_tgt', out_file=train_file, args=args, line_check=False
            )
        self._pre_process_parallel(
            'valid_src', 'valid_tgt', out_file=self.valid_file, args=args, line_check=False
        )

        if args.get("finetune_src") or args.get("finetune_tgt"):
            self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)

        self.persist_state()
        self._prepared_flag.touch()

    def inherit_parent(self):
        parent = self.config['parent']
        if parent.get('shrink'):
            raise ValueError('parent.shrink not supported for this model yet')
        super(ClassificationExperiment, self).inherit_parent()

    def get_predictions(
        self, model, input: Union[str, Path, List[str]], batch_size: Union[int, Tuple[int, int]], max_len=256
    ):
        """
        :param model:
        :param input: either a path string or Path object, or list of strings
        :param batch_size:
        :param max_len:
        :return:
        """
        if isinstance(input, (str, Path)):
            texts = IO.get_lines(input)
        else:
            assert isinstance(input, list) and isinstance(input[0], str)
            texts = input
        txt_to_ids = partial(self.src_field.encode_as_ids, add_bos=False, add_eos=True)
        texts = (txt_to_ids(x)[:max_len] for x in texts)
        # sort as descending order of lengths
        texts_lensorted = list(sorted(enumerate(texts), key=lambda x: len(x[1]), reverse=True))
        log.info(
            f"Predicting labels for {len(texts_lensorted)} sentences;"
            f" batch_size={batch_size} max_len={max_len}"
        )
        model = model.eval().to(device)
        preds = []
        top1_probs = []
        pad_idx = self.src_field.pad_idx

        def _consume_minibatch(buffer):
            nonlocal preds, top1_probs, model, pad_idx  # accessing outer variable
            max_len = max(len(x) for orig_i, x in buffer)
            x_seqs = torch.full((len(buffer), max_len), fill_value=pad_idx, dtype=torch.long)
            batch_is = [batch_i for batch_i, x in buffer]
            for minibatch_i, (batch_i, x) in enumerate(buffer):
                x_seqs[minibatch_i, : len(x)] = torch.tensor(x, dtype=torch.long)

            x_seqs = x_seqs.to(device)
            x_mask = (x_seqs != pad_idx).unsqueeze(1)
            probs = model(src=x_seqs, src_mask=x_mask, score='softmax')
            top_1probs, top_1 = probs.max(dim=1)

            preds += list(zip(batch_is, top_1.tolist()))
            top1_probs += list(zip(batch_is, top_1probs.tolist()))

        if isinstance(batch_size, int):
            max_toks, max_sents = batch_size, float('inf')
        else:
            max_toks, max_sents = batch_size

        buffer = []
        tok_count = 0
        with tqdm.tqdm(texts_lensorted, total=len(texts_lensorted)) as data_bar:
            for idx, txt in data_bar:
                buffer.append((idx, txt))
                tok_count += len(txt)
                if tok_count >= max_toks or len(buffer) >= max_sents:
                    _consume_minibatch(buffer)
                    # new batch
                    buffer.clear()
                    tok_count = 0
            if buffer:
                _consume_minibatch(buffer)

        # restore order, drop indices
        preds_idx = [p for i, p in sorted(preds, key=lambda x: x[0])]
        pred_labels = [self.tgt_vocab.class_names[idx] for idx in preds_idx]
        top1_probs = [p for i, p in sorted(top1_probs, key=lambda x: x[0])]
        return preds_idx, pred_labels, top1_probs

    def evaluate_classifier(self, model, input: Path, labels: Path, batch_size, max_len: int):
        model = model.eval()
        pred_idx, pred_labels, probs = self.get_predictions(
            model, input, batch_size=batch_size, max_len=max_len
        )
        label_to_id = partial(self.tgt_field.encode_as_ids, add_bos=False, add_eos=False)
        labels = [label_to_id(x)[0] for x in IO.get_lines(labels)]
        assert len(pred_idx) == len(labels), f'preds:{len(pred_idx)} == truth:{len(labels)}?'
        log.info(f"Testing on {len(labels)} examples")
        clsmap = self.tgt_field.class_names
        metric = ClsMetric(prediction=pred_idx, truth=labels, clsmap=clsmap)
        return metric, pred_labels, probs
