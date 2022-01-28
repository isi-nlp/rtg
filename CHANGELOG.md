# v0.6.1 : WIP
- `rtg.fork` accepts multiple to_dir; thus supports cloning multiple times at once
- Bug fix: early stopping on distributed parallel training
- `rtg.tool.augment` to support data augmentations
- Add attention visualization in rtg.serve; powered by plotly
- rtg.pipeline and rtg.fork: uses relative symlinks instead of absolute paths
- rtg.decode shows decoding speed (segs, src_toks, hyp_toks)
- `batch_size` is auto adjusted based on number of workers and gradient_accum (huh! finally)
- `batch_size` normalizer in distributed training setting (fix! faster convergence now)
- support for `byte` encoding added
- 

# v0.6.0 : 20210921
- Redesign of registry; using decorators to register all modules
- `optim` block is split into `optimizer` `schedule` and `criterion; as a result, **this version is not backward compatible with prior versions** Refer to migration guide
  - `NoamOpt` replaced with `ScheduledOptimizer` which takes scheduler and optimizer objects which are independently configurable from conf.yml
    
- Add transformer sequence classification model: `tfmcls`, supports initialization from pretrained NMT (picks encoder layers, source embeddings, and source vocabs from NMT experiment)

# 0.5.2 : 20210821
- Fix `rtg.decode` bug fix (partial migration to new API)
  - test case added for `decode` api so we can catch such errors in future

# v0.5.1 : 20210814
- Add `rtg-params` command that shows trainable parameters in model (layer wise as well as total)
- TextTransform moved inside Experiment
  - Integrated into rtg.pipeline as well as into validation metrics
- validation on detokenized bleu, chrf, etc is now supported 
  - `valid_tgt_raw` is now required
- Criterion:
  - Sparse and Dense CrossEntropy
    - Weighted Cross Entropy, with label smoothing
  - Dice Loss (WIP)
  - Squared Error
- vocab management
  - `prep.pieces` originally took a string, now it can take either a `string` (i.e. same scheme for both source and target) or
   `[string, string]` separate scheme for source and target when `shared=false`.
  - Example: `pieces: [char, bpe]` with `shared: false` makes char pieces on source side and `bpe` pieces on target   


# v0.5.1 : 20210814
- `rtg.serve` supports flexible transformations on source (pre processing) and target (post processing)
- Travis build configured to auto run tests
- sequence classification is now supported via `tfmcls` model

# v0.5.0 : 20210329
- DDP: multinode training see `scripts/slurm-multinode-launch.sh`
- FP16 and mixed precision (upgrade from APEX to torch's built in AMP)
- NLCodec & NLDb integration for scaling to large datasets using pyspark backend
- Web UI rtg-serve
- Cache ensemble state for rtg-decode
- Docker images for 500-eng model
- Parent child transfer: Shrink parent model vocab and embeddings to child datasets 
- Fix packaging of flask app: now templates and static files are also included in PyPI package


# v0.4.2 : 20200824 
- Fix issue with dec_bos_cut complaining that tensors are not on contigous storage
- REST API
- Docker for deployment
