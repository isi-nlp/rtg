# v0.5.1 : WIP

- Add `rtg-params` command that shows trainable parameters in model (layer wise as well as total)
- `rtg.serve` supports flexible transformations on source (pre processing) and target (post processing)
- Travis build configured to auto run tests 


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
