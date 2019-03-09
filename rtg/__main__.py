import sys
from rtg import log, __version__, RTG_PATH

if __name__ == '__main__':
    log.info(f'RTG_PATH={RTG_PATH}')
    log.info(f'version={__version__}')
    log.error('please use rtg.pipeline, rtg.prep, rtg.train or rtg.decode sub tasks')
    sys.exit(1)
