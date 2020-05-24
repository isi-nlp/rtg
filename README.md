# Reader-Translator-Generator (RTG)  

Reader-Translator-Generator (RTG) is a Neural Machine Translation toolkit based on pytorch. 

## Features
- Reproducible experiments: one `conf.yml`  that has everything -- data paths, params, and
   hyper params -- required to reproduce experiments.
- Pre-processing options: [sentencepiece](https://github.com/google/sentencepiece) or [nlcodec](https://github.com/isi-nlp/nlcodec) (or add your own) 
    -  word/char/bpe etc types
    - shared vocabulary, seperate vocabulary
    - one-way, two-way, three-way tied embeddings
- [Transformer model from "Attention is all you need"](https://arxiv.org/abs/1706.03762) (fully tested and competes with [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
   - Automatically detects and parallelizes across multi GPUs. (Note: All GPUs must be in the same node, though!)
   - Lot of varieties of transformer: width varying, skip transformer etc  
- [RNN based Encoder-Decoder](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) with [Attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf) . (No longer use it, but it's there for experimentation)
- Language Modeling: RNN, Transformer
- And more ..
  + Easy and interpretable code (for those who read code as much as papers)
  + Object Orientated Design. (Not too many levels of functions and function factories like Tensor2Tensor)
  + Experiments and reproducibility are main focus. To control an experiment you edit an YAML file that is inside the experiment directory.
  + Where ever possible, prefer [convention-over-configuation](https://www.wikiwand.com/en/Convention_over_configuration). Have a look at this experiment directory for the [examples/transformer.test.yml](examples/transformer.test.yml);

### Setup
Add the root of this repo to `PYTHONPATH` or install it via `pip --editable`

```bash
git clone https://github.com/isi-nlp/rtg.git 
cd rtg                # go to the code
# use https://github.com/isi-nlp/rtg-xt.git if you dont have access to rtg.git

conda env create -n rtg python=3.7.   # adds a conda env named rtg
conda activate rtg  # activate it

# install this as a local editable pip package
pip install --editable .   
# The requirements are in setup.py

# or add it to PYTHONPATH 
export PYTHONPATH=$PWD 
```

# Usage

Refer to `scripts/rtg-pipeline.sh` bash script and `examples/transformer.base.yml` file

```bash
# use examples/transformer.base.yml config to setup an experiment at 001-tfm dir (TODO: edit paths in yml file)

# if you have rtg module in $PYTHONPATH (manula export or via pip install)
$ python -m rtg.pipeline 001-tfm examples/transformer.base.yml

# or if your have `rtg-pipeline` command in $PATH (via pip install) 
rtg-pipe 001-tfm examples/transformer.base.yml
```

The `001-tfm` directory that hosts an experiment looks like this:
```
001-tfm
├── _PREPARED    <-- Flag file indicating experiment is prepared 
├── _TRAINED     <-- Flag file indicating experiment is trained
├── conf.yml     <-- Where all the params and hyper params are! You should look into this
├── data        
│   ├── samples.tsv.gz          <-- samples to log after each check point during training
│   ├── sentpiece.shared.model  <-- as the name says, sentence piece model, shared
│   ├── sentpiece.shared.vocab  <-- as the name says
│   ├── train.db                <-- all the prepared trainig data in a sqlite db
│   └── valid.tsv.gz            <-- and the validation data
├── githead       <-- whats was the git HEAD hash this experiment was started? 
├── job.sh.bak    <-- job script used to submit this to grid. Just in case
├── models        <-- All checkpoints go inside this
│   ├── model_400_5.265583_4.977106.pkl
│   ├── model_800_4.478784_4.606745.pkl
│   ├── ...
│   └── scores.tsv <-- train and validation losses. incase you dont want to see tensorboard
├── rtg.log   <-- the python logs are redirected here
├── rtg.zip   <-- the source code used to run. just `export PYTHONPATH=rtg.zip` to 
├── scripts -> /Users/tg/work/me/rtg/scripts  <-- link to some perl scripts for detok+BLEU
├── tensorboard    <-- Tensorboard stuff for visualizations
│   ├── events.out.tfevents.1552850552.hackb0x2
│   └── ....
└── test_step2000_beam4_ens5   <-- Tests after the end of training, BLEU scores
    ├── valid.ref -> /Users/tg/work/me/rtg/data/valid.ref
    ├── valid.src -> /Users/tg/work/me/rtg/data/valid.src
    ├── valid.out.tsv
    ├── valid.out.tsv.detok.tc.bleu
    └── valid.out.tsv.detok.lc.bleu

```

# Docs

Refer to [docs](./docs) directory, and especially documentation on [conf.yml](./docs/conf.yml.adoc) 
where the crucial information is.


---------
### Authors:
[See Here](https://github.com/thammegowda/rtg/graphs/contributors)


### Credits / Thanks
+ OpenNMT and the Harvard NLP team for [Annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html), I learned a lot from their work
+ [My team at USC ISI](https://www.isi.edu/research_groups/nlg/people) for everything else