#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/22/20
from rtg.module.trainer import EarlyStopper


def test_is_stop():

    losses = [10, 9, 8, 7, 6, 5, 6, 5, 6, 7, 8, 8, 6, 5, 6, 7, 6]
    stopper = EarlyStopper(patience=5, by='loss')
    stopped = False
    for i, val in enumerate(losses):
        stopper.step()
        stopper.validation(val)
        if stopper.is_stop():
            stopped = True
            break
    assert stopped

