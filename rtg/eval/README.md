# rtg.eval

This module is for evaluation tools.
(Currently not much is there, however, overtime the desired tools will be accumulated)


## Compute BLEU+1 (i.e. BLEU per line)

```
$ python -m rtg.eval.linebleu -h
usage: linebleu.py [-h] [-c CANDS] [-r REFS] [-n N] [-o OUT] [-v]


Computes BLEU score per record.

optional arguments:
  -h, --help            show this help message and exit
  -c CANDS, --cands CANDS
                        Candidate (aka output from NLG system) file (default:
                        <_io.TextIOWrapper name='<stdin>' mode='r'
                        encoding='UTF-8'>)
  -r REFS, --refs REFS  Reference (aka human label) file (default:
                        <_io.TextIOWrapper name='<stdin>' mode='r'
                        encoding='UTF-8'>)
  -n N, --n N           maximum n as in ngram. (default: 4)
  -o OUT, --out OUT     Output file path to store the result. (default:
                        <_io.TextIOWrapper name='<stdout>' mode='w'
                        encoding='UTF-8'>)
  -v, --verbose         verbose mode (default: False)
```


## (log) Perplexity of a language model:

```
$ python -m rtg.eval.perplexity  -h
usage: rtg.eval.perplexity [-h] [-t TEST] [-en ENSEMBLE]
                           work_dir [model_path [model_path ...]]

positional arguments:
  work_dir              Working/Experiment directory
  model_path            Path to model's checkpoint. If not specified, a best
                        model (based on the score on validation set) from the
                        experiment directory will be used. If multiple paths
                        are specified, then an ensembling is performed by
                        averaging the param weights (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -t TEST, --test TEST  test file path. default is STDIN (default:
                        <_io.TextIOWrapper name='<stdin>' mode='r'
                        encoding='UTF-8'>)
  -en ENSEMBLE, --ensemble ENSEMBLE
                        Ensemble best --ensemble models by averaging them
                        (default: 1)
```
