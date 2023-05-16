import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from rtg import Batch, TSVData, device, line_count, log
from rtg.classifier import ClassificationExperiment
from rtg.data.codec import Field as BaseField
from rtg.comet import HFField, Example, Batch


class HfTransformerExperiment(ClassificationExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = self.model_args['model_id']
        assert self.model_id.startswith('hf:'), 'only huggingface models are supported'
        self.model_id = self.model_id[3:]
        self.src_field = HFField(self.model_id)

    def pre_process(self, args=None, force=False):
        args = args or self.config.get('prep')
        log.info(f"Pre-processing data for {self.model_id}")
        # TODO: make the src vocab load from experiment dir (i.e., offline use)
        if force or not self._tgt_field_file.exists():
            # target vocabulary; class names. treat each line as a word
            tgt_corpus = []
            if args.get('train_tgt') and not args.get('train_tgt').startswith('stdin:'):
                tgt_corpus.append(args['train_tgt'])
            if args.get('mono_tgt'):
                tgt_corpus.append(args['mono_tgt'])
            assert tgt_corpus, 'prep.train_tgt (not stdin) or prep.mono_tgt must be defined'
            # NLCodec Class Field
            self.tgt_field = self._make_vocab(
                "tgt", self._tgt_field_file, 'class', corpus=tgt_corpus, vocab_size=-1
            )
        n_classes = self.config['model_args'].get('tgt_vocab')
        if len(self.tgt_field) != n_classes:
            log.warning(
                f'model_args.tgt_vocab={n_classes},' f' but found {len(self.tgt_field)} cls in {tgt_corpus}'
            )
        self._pre_process_parallel(
            'valid_src', 'valid_tgt', out_file=self.valid_file, args=args, line_check=False
        )

        self.persist_state()
        # self._prepared_flag.touch()

    def get_train_data(
        self,
        batch_size: Union[int, Tuple[int, int]],
        **kwargs,
    ):
        if kwargs:
            log.warning(f'Ignoring kwargs: {kwargs}')
        train_src = self.config.get('prep', {}).get('train_src', '').lower()
        train_tgt = self.config.get('prep', {}).get('train_tgt', '').lower()
        if train_src.startswith('stdin:') and train_tgt.startswith('stdin:'):
            # TODO: implement :{raw/bin}:idx for stdin
            log.info(f'==Reading train data from stdin==')
            ex_stream = self.stream_line_to_example(sys.stdin, **self._get_batch_args())
        else:
            assert self.train_db.exists()
            from nlcodec.db import MultipartDb

            ex_stream = MultipartDb.load(self.train_db, shuffle=True, rec_type=Example)

        fields = [self.src_field, self.src_field, self.tgt_vocab]
        batch_stream = self.stream_example_to_batch(
            ex_stream, batch_size, fields=fields, **self._get_batch_args()
        )
        return batch_stream

    def _input_line_encoder(self, line: str):
        cols = line.split('\t')
        assert len(cols) >= 2
        res = [self.src_field.encode_as_ids(col) for col in cols[:2]]
        if len(cols) > 2:  # if there is a label in third col
            res.append(self.tgt_vocab.encode_as_ids(cols[2]))
        return tuple(res)

    def _pre_process_parallel(
        self,
        src_key: str,
        tgt_key: str,
        out_file: Path,
        args: Optional[Dict[str, Any]] = None,
        line_check=True,
    ):
        """
        Pre process records of a parallel corpus
        :param args: all arguments for 'prep' task
        :param src_key: key that contains source sequences
        :param tgt_key: key that contains target sequences
        :param out_file: path to store processed TSV data (compresses if name ends with .gz)
        :return:
        """
        args = args if args else self.config['prep']
        assert src_key in args, f'{src_key} not found in experiment config or args'
        assert tgt_key in args, f'{tgt_key} not found in experiment config or args'
        src_file = args[src_key]
        tgt_file = args[tgt_key]
        if 'stdin:' in src_file or 'stdin:' in tgt_file:
            log.info('Skipping prep since data is from stdin')
            return

        if line_check:
            assert line_count(args[src_key]) == line_count(
                args[tgt_key]
            ), f'{args[src_key]} and {args[tgt_key]} must have same number of lines'

        log.info(f"Going to prep files {src_key} and {tgt_key}")
        s_time = time.time()
        reader_func = TSVData.read_raw_parallel_recs
        parallel_recs = reader_func(
            src_file,
            tgt_file,
            args['truncate'],
            args['src_len'],
            args['tgt_len'],
            src_tokenizer=self._input_line_encoder,
            tgt_tokenizer=partial(self.tgt_vocab.encode_as_ids),
        )
        parallel_recs = ((s[0], s[1], t) for s, t in parallel_recs)  # flatten the tuple

        if any([out_file.name.endswith(suf) for suf in ('.nldb', '.nldb.tmp')]):
            from nlcodec.db import MultipartDb

            MultipartDb.create(path=out_file, recs=parallel_recs, field_names=('x1', 'x2', 'y'))
        else:
            TSVData.write_parallel_recs(parallel_recs, out_file)
        e_time = time.time()
        log.info(f"Time taken to process: {timedelta(seconds=(e_time - s_time))}")

    def stream_line_to_example(
        self, stream: Iterator[str], max_src_len: int = 512, max_tgt_len: int = 512, truncate=True
    ) -> Iterator[Example]:
        for idx, line in enumerate(stream):
            row = self._input_line_encoder(line)
            assert len(row) == 3, f'Expected 3 columns, but found {len(row)}'
            x1, x2, y = row[:3]
            if truncate:
                x1, x2, y = x1[:max_src_len], x2[:max_src_len], y[:max_tgt_len]
            elif len(x1) > max_src_len or len(x2) > max_src_len or len(y) > max_tgt_len:
                # Skipping line with length > max_len. Current idx: {idx}
                self.n_skips += 1
                continue
            yield Example(id=idx, x1=x1, x2=x2, y=y)
        log.warning('StreamData is exhausted')

    def stream_example_to_batch(
        self,
        stream: Iterator[Example],
        batch_size: Union[int, Tuple[int, int]],
        fields: List[BaseField],
        device=device,
        **kwargs,
    ) -> Iterator[Batch]:
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of tokens on the target size per batch
        :param raw_path: (src, tgt) paths for loading the sentences (optional); use it for validation
               required: keep_mem=true, shuffle=False, sort_by=None
        :param keep_in_mem: keep the dataset in-memory
        :param sort_desc: should the batch be sorted by src sequence len (useful for RNN api)
        """

        if isinstance(batch_size, int):
            max_toks, max_sents = batch_size, batch_size
        else:
            max_toks, max_sents = batch_size

        batch = []
        max_len = 0
        for ex in stream:
            if min(len(ex.x1), len(ex.x2), len(ex.y)) == 0:
                log.warn("Skipping a record,  either source or target is empty")
                continue

            this_len = max(len(ex.x1), len(ex.x2), len(ex.y))
            if len(batch) < max_sents and (len(batch) + 1) * max(max_len, this_len) <= max_toks:
                batch.append(ex)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > max_toks:
                    log.warn(
                        f'Unable to make a batch of {max_toks} toks'
                        f' with a seq of x1:{len(ex.x1)} x2:{len(ex.x2)} y:{len(ex.y)}'
                    )
                    continue
                # yield the current batch
                yield Batch(batch, fields=fields, device=device)
                batch = [ex]  # new batch
                max_len = this_len
        if batch:
            log.debug(f"\nLast batch, size={len(batch)}")
            yield Batch(batch, fields=fields, device=device)
