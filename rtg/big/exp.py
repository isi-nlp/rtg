#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/7/20
import math
import time
from datetime import timedelta
from torch.multiprocessing import set_start_method, Process, Queue
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple, Iterator, Iterable

import pyspark
import pyspark.sql.functions as SF
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, LongType

from rtg import cpu_count, log
from rtg.data.dataset import Batch, IdExample
from rtg.exp import TranslationExperiment, Field

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def get_spark_session(config: Dict[str, str]) -> SparkSession:
    """
    :param config: dict of key:value pairs for spark
    :return:
    """
    log.info("Creating or restoring a spark session")
    builder = SparkSession.builder
    for k, v in config.items():
        log.info(f"{k}={v}")
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    ui_url = spark.sparkContext.uiWebUrl
    log.info(f"You may access spark web UI at: {ui_url}")
    return spark


class BigTranslationExperiment(TranslationExperiment):

    def __init__(self, work_dir: Union[str, Path], read_only=False,
                 config: Union[str, Path, Optional[Dict[str, Any]]] = None):
        super().__init__(work_dir=work_dir, read_only=read_only, config=config)
        assert self.codec_name == 'nlcodec', 'only nlcodec is supported for big experiments'
        self.train_file = self.data_dir / "train.parquet"
        self.train_db = self.data_dir / "train.parquet"
        self.finetune_file = self.data_dir / "finetune.parquet"
        self._spark = None
        assert 'spark' in self.config, 'refer to docs for enabling spark backend'
        spark_conf = self.config['spark']
        if 'spark.master' not in spark_conf:
            spark_conf['spark.master'] = f'local[{cpu_count}]'
        self.len_sort_size = spark_conf.get('len_sort_size', 20_000)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_spark"]  # Don't pickle _spark
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # _spark gets auto created lazily

    def spark_session(self):
        if not self._spark:
            self._spark = get_spark_session(self.config['spark'])
        return self._spark

    def _pre_process_parallel(self, src_key: str, tgt_key: str, out_file: Path,
                              args: Optional[Dict[str, Any]] = None, line_check=False):
        """
        Pre process records of a parallel corpus
        :param args: all arguments for 'prep' task
        :param src_key: key that contains source sequences
        :param tgt_key: key that contains target sequences
        :param out_file: path to store processed TSV data (compresses if name ends with .gz)
        :return:
        """
        if not out_file.name.endswith(".parquet"):
            if 'train' in out_file.name:
                log.warning(f"set  .parquet extension to enable spark")
            return super()._pre_process_parallel(
                src_key=src_key, tgt_key=tgt_key, out_file=out_file, args=args,
                line_check=line_check)

        args = args if args else self.config['prep']
        log.info(f"Going to prep files {src_key} and {tgt_key}")
        assert src_key in args, f'{src_key} not found in experiment config or args'
        assert tgt_key in args, f'{tgt_key} not found in experiment config or args'

        # create Piece IDs
        s_time = time.time()
        spark = self.spark_session()
        df = SparkDataset.read_raw_parallel_recs(
            spark, args[src_key], args[tgt_key], args['truncate'], args['src_len'], args['tgt_len'],
            src_tokenizer=self.src_vocab.encode_as_ids, tgt_tokenizer=self.tgt_vocab.encode_as_ids)
        log.warning(f"Storing data at {out_file}")
        df.write.parquet(str(out_file))
        e_time = time.time()
        log.info(f"Time taken to process: {timedelta(seconds=(e_time - s_time))}")

    def _make_vocab(self, name: str, vocab_file: Path, model_type: str, vocab_size: int,
                    corpus: List, no_split_toks: List[str] = None, char_coverage=0) -> Field:
        spark = self.spark_session()

        if vocab_file.exists():
            log.info(f"{vocab_file} exists. Skipping the {name} vocab creation")
            return self.Field(str(vocab_file))
        flat_uniq_corpus = set()  # remove dupes, flat the nested list or sets
        for i in corpus:
            if isinstance(i, set) or isinstance(i, list):
                flat_uniq_corpus.update(i)
            else:
                flat_uniq_corpus.add(i)

        flat_uniq_corpus = list(flat_uniq_corpus)
        log.info(f"Going to build {name} vocab from {len(flat_uniq_corpus)} files ")
        return self.Field.train(model_type, vocab_size, str(vocab_file), flat_uniq_corpus,
                                no_split_toks=no_split_toks, char_coverage=char_coverage,
                                spark=spark)

    def get_train_data(self, batch_size: int, steps: int = 0, sort_by='eq_len_rand_batch',
                       batch_first=True, shuffle=False, fine_tune=False, keep_in_mem=False):
        data_file: Path = self.train_file
        if fine_tune:
            if not self.finetune_file.exists():
                # user may have added fine tune file later
                self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)
            log.info("Using Fine tuning corpus instead of training corpus")
            data_file = self.finetune_file
        assert data_file.name.endswith(".parquet")
        assert not keep_in_mem, 'keep in memory not supported for big experiments'
        # data_file = IO.maybe_tmpfs(data_file)

        # Read data on a separate process
        # all these args should be easily pickle-able
        max_q_size = 10 ** 4
        queue = Queue(max_q_size)
        batching_args = dict(batch_size=batch_size, sort_by=sort_by,
                             batch_first=batch_first, shuffle=shuffle,
                             field=self.tgt_vocab, buffer_size=max_q_size)
        batching_args.update(self._get_batch_args())

        prod_args = dict(queue=queue, total=steps, spark_conf=self.config['spark'],
                         data_path=data_file, batching_args=batching_args)
        prod_proc = Process(target=producer, kwargs=prod_args)
        prod_proc.daemon = True  # according to docs, all daemon children be killed when parent exits
        prod_proc.start()
        log.info(f"Started a separate process to read dataset: PID: {prod_proc.pid}")

        def consumer(queue):
            while True:
                item = queue.get()
                if item == 'ERROR':
                    raise Exception("Data producer failed with some error. Check logs")
                if item == 'DONE':
                    break
                yield item

        # prod_proc.join()  # the consumer() waits for 'DONE' message that is analogous to join()
        return consumer(queue)


def producer(queue: Queue, total: int, spark_conf: Dict, data_path: Union[str, Path],
             batching_args: Dict):
    """

    :param queue: queue for communicating data
    :param total: how many batches to read. Keeps looping if total > data; or aborts if data < steps
    :param spark_conf: spark configuration to create or restore session on a new process
    :param data_path:  path to dataset
    :param batching_args: args to `SparkDataset`
    :return:
    """
    try:
        assert total > 0
        spark = get_spark_session(spark_conf)
        if not isinstance(data_path, str):
            data_path = str(data_path)
        data = spark.read.parquet(data_path)
        dataset = SparkDataset(data, **batching_args)
        count = 0
        epochs = 0
        while count < total:
            epochs += 1
            for batch in dataset:
                queue.put(batch, block=True)
                count += 1
                if count >= total:
                    break
        log.info(f"Production ending; steps={count} epochs={epochs} ")
        queue.put('DONE')  # All good
    except:
        queue.put('ERROR')  # communicate to child so it can decide to die
        raise  # and die


class IncompleteBatchException(Exception):

    def __init__(self, batch: List):
        self.batch = batch


class SparkDataset(Iterable[Batch]):

    # This should have been called as Dataset
    def __init__(self, data: DataFrame, batch_size: int, field: Field,
                 sort_desc: bool = False, batch_first: bool = True,
                 sort_by: str = None, buffer_size=20_000, shuffle=True, device=cpu_device,
                 max_src_len: int = 512, max_tgt_len: int = 512, truncate: bool = False):
        """
        :param data: Dataframe
        :param batch_size: batch size in tokens
        :param field: an instance of Field, to access <bos> <eos> <pad> indexes
        :param sort_desc: sort mini batch by descending order of length (for RNNs)
        :param batch_first:  batch is the first dimension
        :param sort_by:supported:  None or eq_len_rand_batch .
        :param buffer_size: buffer size if sort_by=eq_len_rand_batch
        :param shuffle: shuffle datasets between epochs
        :param max_src_len:
        :param max_tgt_len:
        :param truncate: True => truncate src and tgt sequences by  max_src_len and max_tgt_len .
            False => drop sequence if either  len(src) > max_src_len or len(tgt) > max_tgt_len
        """
        assert isinstance(data, DataFrame), 'data must be of type pyspark.sql.DataFrame'
        self.n_epoch = 0
        self.field = field
        self.sort_desc = sort_desc
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.sort_by = sort_by

        self.data = data.persist(pyspark.StorageLevel.MEMORY_AND_DISK)  # .cache()
        self._n_rows = self.data.select('id').count()
        self.buffer_size = buffer_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.truncate = truncate
        self.shuffle = shuffle
        self.device = device
        log.info(f'Batch Size = {batch_size} toks, sort_by={sort_by}; total_rows={self._n_rows}')

    def make_batches(self, data, raise_incomplete=False) -> Iterator[Batch]:
        """
        :param data: iterator of data
        :param raise_incomplete:
        :return:
        """
        batch = []
        max_len = 0
        for ex in data:
            if min(len(ex.x), len(ex.y)) == 0:
                log.warn("Skipping a record,  either source or target is empty")
                continue

            this_len = max(len(ex.x), len(ex.y))
            if (len(batch) + 1) * max(max_len, this_len) <= self.batch_size:
                batch.append(ex)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.batch_size:
                    raise Exception(f'Unable to make a batch of {self.batch_size} toks'
                                    f' with a seq of x_len:{len(ex.x)} y_len:{len(ex.y)}')
                # yield the current batch
                yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                            field=self.field, device=self.device)
                batch = [ex]  # new batch
                max_len = this_len
        if batch:
            if raise_incomplete:  # throw back to caller
                raise IncompleteBatchException(batch=batch)
            else:
                log.debug(f"\nLast batch, size={len(batch)}")
                yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                            field=self.field, device=self.device)
        # else all items are consumed

    def read_all(self, data):
        return self.make_batches(data=self.to_local(data), raise_incomplete=False)

    def to_local(self, data: DataFrame):
        # Note: this is anti-pattern in spark. figure out how to do without localIterator
        for ex in data.rdd.toLocalIterator():
            if len(ex.x) > self.max_src_len or len(ex.y) > self.max_tgt_len:
                if self.truncate:
                    ex.x = ex.x[:self.max_src_len]
                    ex.y = ex.y[:self.max_tgt_len]
                else:  # skip
                    continue
            yield IdExample(x=ex.x, y=ex.y, id=ex.id)

    def make_eq_len_ran_batches(self, buffer_size=None):
        # every pass introduces some randomness
        data_shuf = self.data.orderBy(SF.rand())
        buffer_size = buffer_size or self.buffer_size
        assert buffer_size > 0
        buffer = []
        n_ex, n_batch = 0, 0
        for ex in self.to_local(data_shuf):
            buffer.append(ex)
            n_ex += 1
            if len(buffer) >= buffer_size:
                # sort
                sorted_buffer = sorted(buffer, key=lambda r: len(r.y))
                try:
                    for b in self.make_batches(data=sorted_buffer, raise_incomplete=True):
                        yield b
                        n_batch += 1
                    buffer = []  # all clear. new buffer
                except IncompleteBatchException as e:
                    buffer = e.batch  # some are left over
        # the last buffer
        if buffer:
            sorted_buffer = sorted(buffer, key=lambda r: len(r.y))
            for b in self.make_batches(data=sorted_buffer, raise_incomplete=False):
                yield b
                n_batch += 1
        log.info(f"rows:{n_ex:,}    batches={n_batch:,}")
        if not n_batch:
            raise Exception(f'Found no training data. Please check conf.yml, especially data paths')

    def __iter__(self) -> Iterator[Batch]:
        self.n_epoch += 1
        if self.sort_by == 'eq_len_rand_batch':
            yield from self.make_eq_len_ran_batches()
        else:
            data = self.data
            if self.shuffle:
                data = data.orderBy(SF.rand())
            yield from self.read_all(data)
        log.info(f"===Epoch {self.n_epoch} completed===")

    @property
    def num_items(self) -> int:
        return self._n_rows

    @property
    def num_batches(self) -> int:
        return int(math.ceil(self._n_rows) / self.batch_size)

    @classmethod
    def read_raw_parallel_recs(cls, spark, src_path: Union[str, Path], tgt_path: Union[str, Path],
                               truncate: bool, src_len: int, tgt_len: int, src_tokenizer,
                               tgt_tokenizer) -> DataFrame:
        raw_df, n_recs = cls.read_bitext(spark, src_path, tgt_path, src_name='x', tgt_name='y')
        tok_rdd = (raw_df.rdd
                   .filter(lambda r: r.x and r.y)  # exclude None
                   .map(lambda r: (r.idx, src_tokenizer(r.x), tgt_tokenizer(r.y)))
                   .filter(lambda r: len(r[1]) and len(r[2]))
                   # exclude empty, if tokenizer created any
                   )
        if truncate:
            tok_rdd = tok_rdd.map(lambda r: (r[0], r[1][:src_len], r[2][:tgt_len]))
        else:
            tok_rdd = tok_rdd.filter(lambda r: len(r[1]) < src_len and len(r[2]) < tgt_len)

        # dataframes doesnt support numpy arrays, so we cast them to python list
        a_row = tok_rdd.take(1)[0]
        if not isinstance(a_row[1], list):
            # looks like np NDArray or torch tensor
            tok_rdd = tok_rdd.map(lambda r: (r[0], r[1].tolist(), r[2].tolist()))
        tok_rdd.map(lambda r: (r[0],))
        df = tok_rdd.toDF(['id', 'x', 'y'])
        return df

    @classmethod
    def read_bitext(cls, spark, src_file: Union[str, Path], tgt_file: Union[str, Path],
                    src_name='src_raw', tgt_name='tgt_raw') -> Tuple[DataFrame, int]:
        if not isinstance(src_file, str):
            src_file = str(src_file)
        if not isinstance(tgt_file, str):
            tgt_file = str(tgt_file)

        src_df = spark.read.text(src_file).withColumnRenamed('value', src_name)
        tgt_df = spark.read.text(tgt_file).withColumnRenamed('value', tgt_name)

        n_src, n_tgt = src_df.count(), tgt_df.count()
        assert n_src == n_tgt, f'{n_src} == {n_tgt} ?'
        log.info(f"Found {n_src:,} parallel records in {src_file, tgt_file}")

        def with_idx(sdf):
            new_schema = StructType(sdf.schema.fields + [StructField("idx", LongType(), False), ])
            return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(
                schema=new_schema)

        src_df = with_idx(src_df)
        tgt_df = with_idx(tgt_df)
        bitext_df = src_df.join(tgt_df, 'idx', "inner")
        # n_bitext = bitext_df.count()
        # assert n_bitext == n_src, f'{n_bitext} == {n_src} ??'
        return bitext_df, n_src
