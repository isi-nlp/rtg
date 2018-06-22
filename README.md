# Yet Another NMT

Yet Another Neural Machine Translation toolkit based on pytorch.
This one hopes to keep code simpler and readable.


### Features / Currently working :
 + Transformer aka Tensor2Tensor or "Attention is all you need"
 + RNN based Encoder-Decoder with Attention (not fully tested, but on the  TODO list)
 + Easy and interpretable (throw the magics out of this box)


### TODO / Near future:
 + Byte Pair Encoding
 + Multi GPU Parallelism
 + RNN model full scale testing
 + Beam decoder


### Setup

```bash
git clone git@github.com:thammegowda/tgnmt.git
cd tgnmt                # go to the code
export PYTHONPATH=$PWD  # Add directory to PYTHONPATH
```

### Quick Start Experiment with Transformer Model

Let us build a translator for xxx --> yyy language

Requirements: A dataset
 + xxx-yyy.train.tok.tsv : Training corpus
 + xxx-yyy.valid.tok.tsv : Validation corpus

Format: TSV format, i.e., a sentence in xxx language and another in yyy language, separated by a tab (`\t`) character.
All tokens should be separated by a regular whitespace character.
Please run a tokenizer on your input. (
[Here is a good one from mosesdecoder](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) repository)


### Step 1. Prepare an experiment

```
$ python -m tgnmt.prep -h
usage: tgnmt.prep [-h] -tf TRAIN_FILE -vf VALID_FILE [-sl SRC_LEN]
                  [-tl TGT_LEN] [-tr]
                  work_dir

prepare NMT experiment

positional arguments:
  work_dir              Working directory

optional arguments:
  -h, --help            show this help message and exit
  -tf TRAIN_FILE, --train-file TRAIN_FILE
                        Training File. (default: None)
  -vf VALID_FILE, --valid-file VALID_FILE
                        Validation File. (default: None)
  -sl SRC_LEN, --src-len SRC_LEN
                        Truncate or filter source sentences to this length
                        (default: 200)
  -tl TGT_LEN, --tgt-len TGT_LEN
                        Truncate or filter target sentences to this length
                        (default: 200)
  -tr, --truncate       Do select all training sentences and truncate them to
                        --src-len and --tgt-len values. Default is to exclude
                        sentences longer than --src-len and --tgt-len
                        (default: False)
```
Example:

```bash
python -m tgnmt.prep work -tf xxx-yyy.train.tok.tsv -vf xxx-yyy.valid.tok.tsv
```
this will create:

```
work/
 +- data/
 |   +- src-field.tsv
 |   +- tgt-field.tsv
 |   +- train.tsv
 |   +- valid.tsv
 +- models/
     +- args.json
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
python -m tgnmt.train work -ne 10 -bs 256
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