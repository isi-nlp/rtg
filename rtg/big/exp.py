#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/7/20
import math
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple, Iterator

import torch
from nlcodec.db.batch import Batch as NBatch, BatchIterable as NBatchIterable, \
    BatchMeta as NBatchMeta
from nlcodec.spark import rdd_as_db, session as spark_session
from nlcodec.utils import log_resources
from pyspark import RDD
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, LongType

from rtg import cpu_count, log
from rtg.data.dataset import Batch, LoopingIterable
from rtg.exp import TranslationExperiment, Field
import numpy as np


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
        self.train_file = self.train_db = self.data_dir / "train.nldb"
        self.finetune_file = self.data_dir / "finetune.nldb"
        self._spark = None
        assert 'spark' in self.config, 'refer to docs for enabling spark backend'
        self.spark_conf = self.config['spark']
        if 'spark.master' not in self.spark_conf:
            self.spark_conf['spark.master'] = f'local[{cpu_count}]'

    def _pre_process_parallel(self, src_key: str, tgt_key: str, out_file: Path,
                              args: Optional[Dict[str, Any]] = None, line_check=False, **kwargs):
        """
        Pre process records of a parallel corpus
        :param args: all arguments for 'prep' task
        :param src_key: key that contains source sequences
        :param tgt_key: key that contains target sequences
        :param out_file: path to store processed TSV data (compresses if name ends with .gz)
        :return:
        """
        if kwargs:
            log.warning(f"The following args are ignored:{kwargs}")
        if not out_file.name.endswith(".nldb"):
            if 'train' in out_file.name:
                log.warning(f"set .nldb extension to enable spark")
            return super()._pre_process_parallel(
                src_key=src_key, tgt_key=tgt_key, out_file=out_file, args=args,
                line_check=line_check)

        args = args if args else self.config['prep']
        log.info(f"Going to prep files {src_key} and {tgt_key}")
        assert src_key in args, f'{src_key} not found in experiment config or args'
        assert tgt_key in args, f'{tgt_key} not found in experiment config or args'

        with log_resources(f"create {out_file.name}"):
            with spark_session(config=self.spark_conf) as spark:
                rdd, total = read_raw_parallel_recs(
                    spark, src_path=args[src_key], tgt_path=args[tgt_key],
                    truncate=args['truncate'], src_len=args['src_len'], tgt_len=args['tgt_len'],
                    src_tokenizer=self.src_vocab.encode_as_ids,
                    tgt_tokenizer=self.tgt_vocab.encode_as_ids)

                id_rdd = rdd.map(lambda r: (r[0], (r[1], r[2])))    # (id, (x, y))
                max_part_size = args.get('max_part_size', 1_000_000)
                n_parts = math.ceil(total / max_part_size)
                log.info(f"Writing to {out_file}; {n_parts} parts,"
                         f" not exceeding {max_part_size:,} records in each part")
                
                rdd_as_db(id_rdd, db_path=out_file, field_names=['x', 'y'], overwrite=True,
                          repartition=n_parts)

    def _make_vocab(self, name: str, vocab_file: Path, model_type: str, vocab_size: int,
                    corpus: List, no_split_toks: List[str] = None, char_coverage=0) -> Field:
        if vocab_file.exists():
            log.info(f"{vocab_file} exists. Skipping the {name} vocab creation")
            return self.Field(str(vocab_file))
        with log_resources(f"create vocab {name}"):
            flat_uniq_corpus = set()  # remove dupes, flat the nested list or sets
            for i in corpus:
                if isinstance(i, set) or isinstance(i, list):
                    flat_uniq_corpus.update(i)
                else:
                    flat_uniq_corpus.add(i)
            with spark_session(config=self.spark_conf) as spark:
                flat_uniq_corpus = list(flat_uniq_corpus)
                log.info(f"Going to build {name} vocab from {len(flat_uniq_corpus)} files ")
                return self.Field.train(model_type, vocab_size, str(vocab_file),
                                        flat_uniq_corpus, no_split_toks=no_split_toks,
                                        char_coverage=char_coverage, spark=spark)

    def get_train_data(self, batch_size: Tuple[int, int], steps: int = 0, sort_by='eq_len_rand_batch',
                       batch_first=True, shuffle=False, fine_tune=False, keep_in_mem=False, **kwargs):
        if kwargs:
            log.warning(f"The following args are ignored:{kwargs}")
        data_path = self.train_file
        if fine_tune:
            if not self.finetune_file.exists():
                # user may have added fine tune file later
                self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)
            log.info("Using Fine tuning corpus instead of training corpus")
            data_path = self.finetune_file

        assert data_path.name.endswith(".nldb")

        vocab = self.tgt_vocab
        batch_meta = NBatchMeta(pad_idx=vocab.pad_idx, bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx,
                                add_bos_x=False, add_bos_y=False, add_eos_x=True, add_eos_y=True)
        
        batches = NBatchIterable(data_path=data_path, batch_size=batch_size, sort_by=sort_by,
                                 batch_first=batch_first, batch_meta=batch_meta)

        data = TLoopingIterable(batches, steps)
        return data


class TBatch(Batch):
    """
    bridging Nlcodec Batch  (numpy) with Batch (torch)
    """

    def __init__(self, batch: NBatch):
        self.batch = batch
        self.meta = meta = batch.meta
        self._len = len(batch)
        self.bos_val: int = meta.bos_idx
        self.eos_val: int = meta.eos_idx
        self.pad_val: int = meta.pad_idx
        self.eos_x = meta.add_eos_x
        self.eos_y = meta.add_eos_y
        self.bos_x = meta.add_bos_x
        self.bos_y = meta.add_bos_y
        self.batch_first = batch.batch_first

        self.x_len = torch.from_numpy(batch.x_len)
        self.x_toks = batch.x_toks
        self.max_x_len = batch.max_x_len
        self.x_seqs = torch.from_numpy(batch.x_seqs)
        self.x_raw = batch.x_raw

        self.has_y = batch.has_y
        if self.has_y:
            self.y_len = torch.from_numpy(batch.y_len)
            self.y_toks = batch.y_toks
            self.max_y_len = batch.max_y_len
            self.y_seqs = torch.from_numpy(batch.y_seqs)
            self.y_raw = batch.y_raw


class TLoopingIterable(LoopingIterable):
    """
    An iterable that keeps looping until a specified number of step count is reached
    """

    def __init__(self, *args, epoch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = epoch

    def __iter__(self) -> Iterator[Batch]:
        while self.count < self.total:
            for batch in self.itr:
                yield TBatch(batch)
                self.count += 1
                if self.count >= self.total:
                    break
            self.epoch += 1
            log.info(f"Epoch {self.epoch} complete.")


def read_raw_parallel_recs(spark, src_path: Union[str, Path], tgt_path: Union[str, Path],
                           truncate: bool, src_len: int, tgt_len: int, src_tokenizer,
                           tgt_tokenizer) -> Tuple[RDD, int]:
    raw_df, n_recs = read_bitext(spark, src_path, tgt_path, src_name='x', tgt_name='y')
    tok_rdd = (raw_df.rdd
               .filter(lambda r: r.x and r.y)  # exclude None
               .map(lambda r: (r.idx, src_tokenizer(r.x), tgt_tokenizer(r.y)))
               .filter(lambda r: len(r[1]) > 0 and len(r[2]) > 0)
               # exclude empty, if tokenizer created any
               )
    if truncate:
        tok_rdd = tok_rdd.map(lambda r: (r[0], r[1][:src_len], r[2][:tgt_len]))
    else:
        tok_rdd = tok_rdd.filter(lambda r: len(r[1]) <= src_len and len(r[2]) <= tgt_len)
    # NOTE: n_recs returned are not exact; filtering may have dropped some
    return tok_rdd, n_recs


def read_bitext(spark, src_file: Union[str, Path], tgt_file: Union[str, Path],
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
