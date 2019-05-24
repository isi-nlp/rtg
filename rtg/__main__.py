import sys
from rtg import log

if __name__ == '__main__':
    log.error('please use rtg.pipeline, rtg.prep, rtg.train or rtg.decode sub tasks')
    sys.exit(1)
