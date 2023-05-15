import sys
from rtg import log

if __name__ == '__main__':
    log.error('please use rtg.cli.pipeline, rtg.cli.prep, rtg.cli.train or rtg.cli.decode sub tasks')
    sys.exit(1)
