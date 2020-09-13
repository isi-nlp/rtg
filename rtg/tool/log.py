#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19
import logging
import os


class Logger(logging.Logger):

    def __init__(self, name='rtg', file=None, file_level=logging.DEBUG, console_level=logging.INFO):
        super().__init__(name, level=logging.DEBUG)
        # create formatter and add it to the handlers
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.formatter = logging.Formatter(
            '[%(asctime)s] p%(process)s {%(module)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%m-%d %H:%M:%S')
        self.console_level = console_level
        self.file_level = file_level

        # create console handler with a higher log level
        self.file = file
        self.fh = None
        self.ch = None
        self.setup_handlers()

    def update_file_handler(self, file, log_level=logging.DEBUG):
        self.file = file
        self.file_level = log_level
        if self.fh and self.fh in self.handlers:
            self.handlers.remove(self.fh)

        path = f'{self.file}-{os.getpid()}'
        self.info(f"Logs are directed to {path}")
        fh = logging.FileHandler(path)
        fh.setLevel(self.file_level)
        fh.setFormatter(self.formatter)
        self.addHandler(fh)
        self.fg = fh

    def setup_handlers(self):
        # create console handler with a higher log level

        self.ch = logging.StreamHandler()
        self.ch.setLevel(self.console_level)
        self.ch.setFormatter(self.formatter)
        self.addHandler(self.ch)

        if self.file:
            self.update_file_handler(self.file, self.file_level)

    def clear_console(self):
        if self.ch and self.ch in self.handlers:
            self.handlers.remove(self.ch)
            self.ch = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['handlers'] = []     # empty this list, dont pickle
        del state["fh"]            # Don't pickle fh
        del state["ch"]            # Don't pickle ch
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.setup_handlers()    # re-setup handlers


