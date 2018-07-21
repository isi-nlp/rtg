# Yet Another NMT

Yet Another Neural Machine Translation toolkit based on pytorch.
>  ** `tgnmt`** is a placeholder.  I will rename it to a presentable name once I have a better alternative.


### Features working  :
 + [Transformer aka Tensor2Tensor or "Attention is all you need"](https://arxiv.org/abs/1706.03762)
 + [RNN based Encoder-Decoder](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) with [Attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
 + [sentencepiece](https://github.com/google/sentencepiece) is under the hood

### Goals:
+ Easy and interpretable code (for those who read code as much as papers)
  + Should be easy to adopt to new settings (the long term goal)
+ Reproducible experiments, based on config files and experiment directory


### TODO :
 + Multi GPU Parallelism
 + Pip installable


### Setup

```bash
git clone git@github.com:thammegowda/tgnmt.git
cd tgnmt                # go to the code
export PYTHONPATH=$PWD  # Add directory to PYTHONPATH
```

### Quick Start Experiment with Transformer Model

Let us build a translator for xxx --> yyy language

Requirements: A dataset
 + train.src, train.tgt : Training corpus
 + valid.src, valid.tgt : Validation corpus

No need to do tokenizer, since the `sentencepiece` library takes care of it under the hood.

### Step 1. Prepare an experiment

```bash
python -m tgnmt.prep work example.conf.yml
```
Where `work` is the diretcory to setup experiment, and `example.conf.yml` shall have these configs
```yaml
src_lang: FRA
tgt_lang: ENG

prep:
  # Training files
  train_src: data/train.src
  train_tgt: data/train.tgt
  valid_src: data/valid.src
  valid_tgt: data/valid.tgt
  src_len: 100
  tgt_len: 100
  truncate: true
  mono_src: []
  mono_tgt: []

# vocabulary
pieces: unigram
vocab_size: 8000

# Testing
test_src:
test_tgt:
```
The `work` directory will have this structure (which you can inspect and modify if need be):

```
work/
 +- data/
 |   +- train.tsv
 |   +- valid.tsv
 |   +- sentpiece.model
 |   +- sentpiece.vocab
 |   +- train.pieces.tsv
 |   +- valid.pieces.tsv
 +- models/
 +- conf.yml
```

### Step 2. Train

```
$ python -m tgnmt.train -h
usage: tgnmt.train [-h] [-mt {rnn,t2t}] [-ne NUM_EPOCHS] [-re]
                   [-bs BATCH_SIZE] [-km KEEP_MODELS]
                   work_dir

Train NMT model

positional arguments:
  work_dir              Working directory

optional arguments:
  -h, --help            show this help message and exit
  -mt {rnn,t2t}, --mod-type {rnn,t2t}
                        Type of model: RNN or T2T (aka transformer) (default:
                        t2t)
  -ne NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Num epochs (default: 15)
  -re, --resume         Resume Training. adds --num-epochs more epochs to the
                        most recent model in work-dir (default: False)
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size (default: 256)
  -km KEEP_MODELS, --keep-models KEEP_MODELS
                        Number of models to keep. Stores one model per epoch
                        (default: 4)
```
Example:

```bash
# for rnn
python -m tgnmt.train work -mt rnn -ne 10 -bs 256
# for transformer
python -m tgnmt.train work -mt t2t -ne 10 -bs 256
```
This step will store last `k` models to `work/models` directory.

### Step 3. Decode

```
$ python -m tgnmt.decode -h
usage: tgnmt.decode [-h] [-if INPUT] [-of OUTPUT] work_dir

Decode using NMT model

positional arguments:
  work_dir              Working directory

optional arguments:
  -h, --help            show this help message and exit
  -if INPUT, --input INPUT
                        Input file path. default is STDIN (default:
                        <_io.TextIOWrapper name='<stdin>' mode='r'
                        encoding='UTF-8'>)
  -of OUTPUT, --output OUTPUT
                        Output File path. default is STDOUT (default:
                        <_io.TextIOWrapper name='<stdout>' mode='w'
                        encoding='UTF-8'>)
```
Exampple:

```bash
$ cat input.tok.txt | python -m tgnmt.decode work > output.tok.txt
```
This step will pick the most recent model from `work/models` directory and translates the input.
