import sys
from tgnmt import log

if __name__ == '__main__':
    log.error('please use tgnmt.prep, tgnmt.train or tgnmt.decode sub tasks')
    sys.exit(1)
