#!/usr/bin/env python
#
#
# Author: Thamme Gowda
# Created: 11/2/21

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Set, List, Sequence

import pyspark
import pyspark.sql.functions as SF
from tqdm import tqdm

from rtg import log
from rtg.big.exp import read_raw_parallel_recs, get_spark_session


@dataclass
class RandomAugment:
    rate: float

    def __post_init__(self):
        assert 0 <= self.rate <= 1

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class RandomWordAugment(RandomAugment):
    def __call__(self, toks):
        assert isinstance(toks, list)
        n_augs = int(len(toks) * self.rate)
        if n_augs > 0:
            aug_pos = {random.randrange(0, len(toks)) for _ in range(n_augs)}
            toks = self.augment(toks, aug_pos)
            return toks
        else:
            return None

    def augment(self, toks: List[Any], positions: Set[int]):
        raise NotImplementedError('Not implemented')


class WordDropout(RandomWordAugment):

    def augment(self, toks: List[Any], positions: Set[int]):
        return [x for i, x in enumerate(toks) if i not in positions]


class WordShuffle(RandomWordAugment):

    def augment(self, toks: List[Any], positions: Set[int]):
        from_pos = list(positions)
        to_pos = [random.randrange(0, len(toks)) for _ in positions]
        for x, y in zip(from_pos, to_pos):
            toks[x], toks[y] = toks[y], toks[x]  # swap
        return toks


@dataclass
class WordReplace(RandomWordAugment):
    replacements: Sequence[Any]

    def augment(self, toks: List[Any], positions: Set[int]):
        for pos in positions:
            toks[pos] = random.choices(self.replacements)
        return toks


@dataclass
class Transforms:
    chain: List[RandomWordAugment]

    def __call__(self, toks):
        x = toks
        for it in self.chain:
            if x is None:
                return None
            x = it(x)
        return x


@dataclass
class Augmentor:
    src_in: Path
    tgt_in: Path
    src_out: Path
    tgt_out: Path
    meta_out: Path
    n_inp_recs: int = 0
    n_out_recs: int = 0
    args: Dict[str, Any] = field(default_factory=dict)
    # ##########
    _spark = None
    _inp_df = None
    _src_freqs = None
    _tgt_freqs = None

    def __post_init__(self):
        paths = [self.src_out, self.tgt_out, self.meta_out]
        for p in paths:
            p.parent.mkdir(exist_ok=True)
        self.outs = [open(p, mode='w', encoding='utf-8') for p in paths]

    def _init_spark(self, config=None):
        config = config or {}
        if 'spark.master' not in config:
            config['spark.master'] = 'local[4]'
        log.info(f"Creating spark with config {config}|")
        self._spark = get_spark_session(config=config)

    @property
    def spark(self):
        if not self._spark:
            self._init_spark()
        return self._spark

    @property
    def inp_df(self):
        if not self._inp_df:
            self._inp_df, self.n_inp_recs = self.read_inp_recs()

            self._inp_df.persist(pyspark.storagelevel.StorageLevel.MEMORY_AND_DISK)
        return self._inp_df

    def read_inp_recs(self):

        def no_op(x):
            return x

        max_len = 2 ** 10
        rdd, n_recs = read_raw_parallel_recs(
            self.spark, src_path=self.src_in, tgt_path=self.tgt_in, truncate=True,
            src_len=max_len, tgt_len=max_len, src_tokenizer=no_op, tgt_tokenizer=no_op)
        df = rdd.toDF(['id', 'src', 'tgt'])
        return df, n_recs

    @property
    def src_freqs(self):
        if not self._src_freqs:
            src_freqs = (self.inp_df.rdd.flatMap(lambda rec: rec[1])
                         .map(lambda tok: (tok, 1))
                         .reduceByKey(lambda a, b: a + b)
                         .collect())
            self._src_freqs = {k: v for k, v in sorted(src_freqs, key=lambda x: x[1], reverse=True)}
        return self._src_freqs

    @property
    def tgt_freqs(self):
        if not self._tgt_freqs:
            tgt_freqs = (self.inp_df.rdd.flatMap(lambda rec: rec[2])
                         .map(lambda tok: (tok, 1))
                         .reduceByKey(lambda a, b: a + b)
                         .collect())
            self._tgt_freqs = {k: v for k, v in sorted(tgt_freqs, key=lambda x: x[1], reverse=True)}
        return self._tgt_freqs

    def write_rec(self, src: str, tgt: str, tag: str):
        self.outs[0].write(f"{src.strip()}\n")
        self.outs[1].write(f"{tgt.strip()}\n")
        self.outs[2].write(f"{tag.strip()}\n")
        self.n_out_recs += 1

    def close(self):
        log.info(f"Written {self.n_out_recs} records to {self.src_out, self.tgt_out}")
        for o in self.outs:
            o.close()

        if self._spark:
            self._spark.stop()

    def run(self, copy=False, noise_src=False, denoise_tgt=False, concat=0, **args):
        config = {
            'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
            'spark.driver.maxResultSize': 0,
        }
        for cli_name, prop_name in [('spark_master', 'spark.master'), ('spark_memory', 'spark.driver.memory'),
                                    ('spark_tmp', 'spark.local.dir')]:
            if args.get(cli_name):
                config[prop_name] = args.get(cli_name)
        self._init_spark(config)
        if copy:
            log.info("copying source to target")
            for rec in self.inp_df.toLocalIterator():
                src, tgt = ' '.join(rec.src), ' '.join(rec.tgt)
                self.write_rec(src, tgt, 'ORIG_INP')
        for enabled, side in [(noise_src, 'src'), (denoise_tgt, 'tgt')]:
            if not enabled:
                continue  # skip
            chain = []
            if args.get('word_drop'):
                chain.append(WordDropout(rate=args.get('word_drop')))
            if args.get('word_shuffle'):
                chain.append(WordShuffle(rate=args.get('word_shuffle')))
            if args.get('word_replace'):
                replacements = list((self.src_freqs if side == 'src' else self.tgt_freqs).keys())
                chain.append(WordReplace(rate=args.get('word_replace'), replacements=replacements))
            if args.get('word_mask'):
                chain.append(WordReplace(rate=args.get('word_replace'), replacements=['<MASK>']))
            transform = Transforms(chain=chain)

            for rec in self.inp_df.toLocalIterator():
                src, tgt = rec.src, rec.tgt
                src, tgt = src.split(), tgt.split()
                if side == 'src':
                    src = transform(src)
                    tag = 'NOISY_SRC'
                else:
                    assert side == 'tgt'
                    src = transform(tgt)
                    tag = 'DENOISE_TGT'
                if not src or not tgt:
                    # this was not augmented, so skip
                    continue
                src, tgt = ' '.join(src), ' '.join(tgt)
                self.write_rec(src, tgt, tag)

        if concat > 0:
            max_src_len = args['max_src_len']
            max_tgt_len = args['max_tgt_len']
            assert max_src_len > 0
            assert max_tgt_len > 0
            df = self.inp_df
            short_df = df.filter((SF.length(df.src) <= int(0.8*max_src_len))
                                 & (SF.length(df.tgt) <= int(0.8*max_tgt_len)))
            short_rdd = short_df.rdd
            cat_rdd = (short_rdd.cartesian(short_rdd)
                       .filter(lambda x: (x[0][0] != x[1][0] and (len(x[0][1]) + len(x[1][1]) <= max_src_len)
                                         and (len(x[0][2]) + len(x[1][2]) <= max_tgt_len))))
                       # .map(lambda x: (x[0][1] + x[1][1], x[0][2] + x[1][2])))
            n_samples = int(self.n_inp_recs * concat)
            # cat_rdd.cache()
            # total_samples = cat_rdd.count()  # this is accurate but too expensive
            total_samples = int(0.5 * self.n_inp_recs * (self.n_inp_recs - 1) * 0.7)  # approximation
            fraction = n_samples / total_samples
            log.info(f"Sampling {self.n_inp_recs:,} x {concat} = {n_samples:,} out of {total_samples:,}"
                     f" total possible concats (approx.). fraction={fraction:g}")
            samples = cat_rdd.sample(withReplacement=False, fraction=fraction)
            i = 0
            for rec1, rec2 in tqdm(samples.toLocalIterator(), desc="Writing concats", total=n_samples):
                src = rec1[1] + ' ' + rec2[1]
                tgt = rec1[2] + ' ' + rec2[2]
                self.write_rec(src, tgt, 'CONCAT1')
                i += 1
                if i >= n_samples:
                    log.info("Aborting early...")
                    break
            if i < n_samples:
                log.warning(f"Expected to get {n_samples} recs but got only {i}")


def main(**kwargs):
    args = kwargs or vars(parse_args())
    props = ['src_in', 'src_out', 'tgt_in', 'tgt_out', 'meta_out']
    props = {pn: args.pop(pn) for pn in props}
    aug = Augmentor(**props)
    try:
        aug.run(**args)
    finally:
        aug.close()


def bound_float(s, mn=0.0, mx=1.0):
    f = float(s)
    assert mn <= f <= mx


def parse_args():
    parser = argparse.ArgumentParser(description="Augment parallel sentences")
    # parser.add_argument("-i", '--inp', type=Path, help="Input file having <source>\\t<target> per line", required=True)
    # parser.add_argument("-o", '--out', type=Path, help="Output file path", required=True)
    io_p = parser.add_argument_group("input-out")
    io_p.add_argument("-si", '--src-in', type=Path, help="Source input file", required=True)
    io_p.add_argument("-so", '--src-out', type=Path, help="Source output file path", required=True)
    io_p.add_argument("-ti", '--tgt-in', type=Path, help="target input file", required=True)
    io_p.add_argument("-to", '--tgt-out', type=Path, help="target output file path", required=True)
    io_p.add_argument("-mo", '--meta-out', type=Path, help="Metadata output file path", required=True)

    copy_p = parser.add_argument_group("copy")
    copy_p.add_argument("-cp", "--copy", action="store_true", help="Copy input to output. default=%(default)s")
    noise_p = parser.add_argument_group("noise")
    noise_p.add_argument("-ns", "--noise-src", action="store_true",
                        help="augment (noise(source), target) records "
                             " See -wd and -ws to control noise rate. default=%(default)s")
    noise_p.add_argument("-dt", "--denoise-tgt", action="store_true",
                        help="Augment (noise(target),target) records."
                             " See -wd and -ws to control noise rate. default=%(default)s")
    noise_p.add_argument("-wd", "--word-drop", metavar="RATE", type=float, default=0.1,
                        help="What percent of words are to be dropped.")
    noise_p.add_argument("-ws", "--word-shuffle", metavar="RATE", type=bound_float, default=0.1,
                        help="What percent of words to shuffle? Range: [0, 1], default=%(default)s")
    noise_p.add_argument("-wr", "--word-random", metavar="RATE", type=bound_float, default=0.1,
                        help="What percent of words to randomly replace? default=%(default)s")

    cat_p = parser.add_argument_group("concatenation")
    cat_p.add_argument("-cat", "--cat", "--concat", dest='concat', metavar='RATIO', type=float, default=0.0,
                        help="RATIO greater than 0 enables random parallel sentence concatenation. default=%(default)s."
                             " Number of augmentations = RATIO*|INPUT|. See -msl and -mtl to control lengths")
    cat_p.add_argument("-msl", "--max-src-len", metavar='N_CHARS', type=int, default=200,
                        help="Maximum chars in source sequence. Active iff -cat > 0. default=%(default)s")
    cat_p.add_argument("-mtl", "--max-tgt-len",  metavar='N_CHARS', type=int, default=200,
                        help="Maximum chars in target sequence. Active iff -cat > 0. default=%(default)s")

    para_p = parser.add_argument_group("parallelization")
    para_p.add_argument("-sm", "--spark-master", type=str, default='local[4]',
                        help="Spark master. set 'local[4]' implies 4 threads on local machine. default=%(default)s")
    para_p.add_argument("-mem", "--spark-memory", metavar='MEM', type=str, default='8g',
                        help="Spark driver memory; higher the better. default=%(default)s")
    para_p.add_argument("-tmp", "--spark-tmp", metavar='PATH', type=str, default=None,
                        help="Spark tmp directory; faster storage is preferred. default=%(default)s")
    return parser.parse_args()


if __name__ == '__main__':
    main()
