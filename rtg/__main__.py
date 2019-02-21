import sys
from rtg import log, __version__

if __name__ == '__main__':
    log.info(f'{__file__} : {__version__}')
    log.error('please use rtg.prep, rtg.train or rtg.decode sub tasks')
    sys.exit(1)
