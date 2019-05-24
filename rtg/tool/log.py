#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19
import logging


class Logger(logging.Logger):

    def __init__(self, name='rtg', file=None, file_level=logging.DEBUG, console_level=logging.INFO):
        super().__init__(name, level=logging.DEBUG)
        # create formatter and add it to the handlers
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.formatter = logging.Formatter(
            '[%(asctime)s] p%(process)s {%(module)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M:%S')
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(self.formatter)
        self.addHandler(ch)

        self.fh = None
        if file:
            fh = logging.FileHandler(file)
            fh.setLevel(file_level)
            fh.setFormatter(self.formatter)
            # add the handlers to the logger
            self.addHandler(fh)
            self.fg = fh

    def update_file_handler(self, file, log_level=logging.DEBUG):
        self.info(f"Logs are directed to {file}")
        if self.fh is not None:
            self.removeHandler(self.fh)
        fh = logging.FileHandler(file)
        fh.setLevel(log_level)
        fh.setFormatter(self.formatter)
        self.addHandler(fh)
