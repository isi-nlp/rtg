# v0.6.0 : WIP

- Redesign of registry; using decorators to register all modules
- `optim` block is split into `optimizer` `schedule` and `criterion; as a result, **this version is not backward compatible with prior versions** Refer to migration guide
  - `NoamOpt` replaced with `ScheduledOptimizer` which takes scheduler and optimizer objects which are independently configurable from conf.yml
    
- Add transformer sequence classification model: `tfmcls`, supports initialization from pretrained NMT (picks encoder layers, source embeddings, and source vocabs from NMT experiment)
- Add `rtg-params` command that shows trainable parameters in model (layer wise as well as total)
 


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
