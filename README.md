# RTG

RTG is a Neural Machine Translation toolkit based on pytorch.
RTG stands for Reader-Translator-Generator when it is noun, and read-translate-generate when it is a verb.
This toolkit is meant for MT/NLG/NLU research.
<small>NOTE: It has no relation to the [regular tree grammar](https://en.wikipedia.org/wiki/Regular_tree_grammar))</small>

### What is working  :
 + [Transformer aka Tensor2Tensor or "Attention is all you need"](https://arxiv.org/abs/1706.03762)
 + [RNN based Encoder-Decoder](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) with [Attention](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
 + [sentencepiece](https://github.com/google/sentencepiece) is under the hood

### Goals:
+ Easy and interpretable code (for those who read code as much as papers)
  + Should be easy to adapt to new settings (the long term goal)
+ Reproducible experiments, based on config files and experiment directory


### TODO :
 + Multi GPU Parallelism (Work in progress)
 + Pip installable with proper versions management


### Known Issues
 + Large Transformer batches cannot be fit into single GPU at the moment, some more simplifications and
 optimizations are coming on the way.   

### Setup

```bash
git clone git@github.com:thammegowda/rtg.git
cd rtg                # go to the code
export PYTHONPATH=$PWD  # Add directory to PYTHONPATH
```

# Usage
This repo is actively under development so whatever I write in README is getting outdated soon.
Refer to `scripts/rtg-pipeline.sh` bash script and `examples/example.*.conf.yml` files

TODO: Write tutorial


---------
### Authors:
[See Here](https://github.com/thammegowda/rtg/graphs/contributors)


### Credits / Thanks
+ OpenNMT and the Harvard NLP team for [Annotated transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html), I learned a lot from their work
+ [My team at USC ISI](https://www.isi.edu/research_groups/nlg/people) for everything else


