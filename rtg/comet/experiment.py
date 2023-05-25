import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from rtg import Batch, TSVData, device, line_count, log, IO
from rtg.classifier import ClassificationExperiment
from rtg.data.codec import Field as BaseField
from rtg.comet import HFField, Example, Batch


class CometExperiment(ClassificationExperiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_src_len = self.config['prep']['src_len']
        self.max_tgt_len = self.config['prep']['tgt_len']
        self.ExampleFactory = partial(
            Example.new_with_length_check, max_src_len=self.max_src_len, max_tgt_len=self.max_tgt_len
        )

    def pre_process(self, args=None, force=False):
        if self._prepared_flag.exists() and not force:
            log.info(f"Pre-processing already done for {self.work_dir}")
            return
        args = args or self.config.get('prep')
        log.info(f"Pre-processing data for {self.model_type}")
        
        if force or not self._src_field_file.exists():
            src_corpus = []
            train_src = args.get('train_src')
            if train_src.startswith('stdin:'):
                src_corpus.append(train_src)
            if args.get('mono_src'):
                src_corpus.append(args['mono_src'])
            assert src_corpus, 'prep.train_src (not stdin) or prep.mono_src must be defined'
            pieces = self.config['prep'].get('pieces', 'bpe')
            src_vocab_size = self.config['prep'].get('max_src_types', self.config['prep'].get('max_types'))
            self.src_field = self._make_vocab(
                "src", self._src_field_file, pieces, corpus=src_corpus, vocab_size=src_vocab_size
            )

        # making tgt vocab from train data
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
                    f'model_args.tgt_vocab={n_classes},'
                    f' but found {len(self.tgt_field)} cls in {tgt_corpus}'
                )
        self._pre_process_parallel(
            'train_src', 'train_tgt', out_file=self.train_db, args=args, line_check=True
        )
        self._pre_process_parallel(
            'valid_src', 'valid_tgt', out_file=self.valid_file, args=args, line_check=True
        )

        self.persist_state()
        self._prepared_flag.touch()

    def get_train_data(
        self,
        steps: int,
        batch_size: Union[int, Tuple[int, int]],
        shuffle=True,
        **kwargs,
    ):
        if kwargs:
            log.warning(f'Ignoring kwargs: {kwargs}')
        train_src = self.config.get('prep', {}).get('train_src', '').lower()
        train_tgt = self.config.get('prep', {}).get('train_tgt', '').lower()

        # read from stdin
        fields = [self.src_field, self.src_field, self.tgt_vocab]
        if train_src.startswith('stdin:') and train_tgt.startswith('stdin:'):
            # TODO: implement :{raw/bin}:idx for stdin
            log.info(f'==Reading train data from stdin==')
            ex_stream = self.stream_line_to_example(sys.stdin, **self._get_batch_args())
            return self.stream_example_to_batch(
                    ex_stream, batch_size, fields=fields, **self._get_batch_args()
                )
        
        # read from file
        assert self.train_db.exists()
        from nlcodec.db import MultipartDb
        assert steps > 0
        def _infinite_stream():
            n_epochs = 0
            count = 0
            while count <= steps:
                n_epochs += 1
                log.info(f'Epoch={n_epochs}; Reading training data from {self.train_db}')
                ex_stream = MultipartDb.load(self.train_db, shuffle=shuffle, rec_type=self.ExampleFactory)
                batch_stream = self.stream_example_to_batch(
                    ex_stream, batch_size, fields=fields, **self._get_batch_args()
                )
                for batch in batch_stream:
                    count += 1
                    yield batch
                    if count > steps:
                        break

        return _infinite_stream()

    def get_val_data(
        self,
        batch_size: Union[int, Tuple[int, int]],
        **kwargs,
    ):
        def read_ex_stream(src_file, tgt_file):
            bargs = dict(
                src_len=self.config['prep']['src_len'],
                tgt_len=self.config['prep']['tgt_len'],
                truncate=self.config['prep']['truncate'],
                src_tokenizer=self._input_line_encoder,
                tgt_tokenizer=partial(self.tgt_vocab.encode_as_ids),
            )
            parallel_recs = TSVData.read_raw_parallel_recs(src_file, tgt_file, **bargs)
            yield from (
                self.ExampleFactory(idx, x1=s[0], x2=s[1], y=t) for idx, (s, t) in enumerate(parallel_recs)
            )

        src_file = IO.resolve(self.config['prep']['valid_src'])
        tgt_file = IO.resolve(self.config['prep']['valid_tgt'])
        ex_stream = read_ex_stream(src_file, tgt_file)
        fields = [self.src_field, self.src_field, self.tgt_vocab]
        batch_stream = self.stream_example_to_batch(
            ex_stream, batch_size, fields=fields, **self._get_batch_args()
        )
        return batch_stream

    def _input_line_encoder(self, line: str):
        cols = line.split('\t')
        assert len(cols) >= 2, f'atleast two column expected, but found {len(cols)}\n{cols}'
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
        parallel_recs = ((s[0], s[1], t) for s, t in tqdm(parallel_recs))  # flatten the tuple

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
            yield self.ExampleFactory(id=idx, x1=x1, x2=x2, y=y)
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



class HFCometExperiment(CometExperiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = self.model_args['model_id']
        assert self.model_id.startswith('hf:'), 'only huggingface models are supported'
        self.model_id = self.model_id[3:]
        self.src_field = HFField(self.model_id)
    
    
    def pre_process(self, args=None, force=False):
        if self._prepared_flag.exists() and not force:
            log.info(f"Pre-processing already done for {self.work_dir}")
            return
        args = args or self.config.get('prep')
        log.info(f"Pre-processing data for {self.model_id}")

        #NOTE:  src vocab should match with pretrained model
            
        # making tgt vocab from train data
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
                    f'model_args.tgt_vocab={n_classes},'
                    f' but found {len(self.tgt_field)} cls in {tgt_corpus}'
                )
        self._pre_process_parallel(
            'train_src', 'train_tgt', out_file=self.train_db, args=args, line_check=True
        )
        self._pre_process_parallel(
            'valid_src', 'valid_tgt', out_file=self.valid_file, args=args, line_check=True
        )

        self.persist_state()
        self._prepared_flag.touch()