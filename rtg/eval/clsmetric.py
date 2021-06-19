#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/18/21

#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 4/26/21

from typing import List, Union
from torch import Tensor
import numpy as np

Array = Union[List[int], Tensor, np.ndarray]


class ClsMetric:

    def __init__(self, prediction: Array, truth: Array, clsmap: List[str]):
        """
        :param prediction: List of predictions (class index)
        :param truth: List of true labels (class index)
        :param clsmap: List of class names for mapping class index to names
        """
        self.clsmap = clsmap
        self.n_classes = len(clsmap)
        self.clsmap_rev = {name: idx for idx, name in enumerate(clsmap)} if clsmap else None
        assert len(prediction) == len(truth)
        assert 0 <= max(prediction) <= self.n_classes
        assert 0 <= max(truth) <= self.n_classes
        self.confusion = self.confusion_matrix(self.n_classes, prediction, truth)
        self.total_gold = self.confusion.sum(axis=1)
        self.total_preds = self.confusion.sum(axis=0)
        assert self.total_preds.sum() == self.total_gold.sum()

        epsilon = 1e-9
        self.correct = self.confusion.diagonal()
        self.precision = 100 * self.correct / (self.total_preds + epsilon)
        self.recall = 100 * self.correct / (self.total_gold + epsilon)
        self.f1 = (2 * self.precision * self.recall / (self.precision + self.recall + epsilon))
        self.col_head = ['Refs', 'Preds', 'Correct', 'Precisn', 'Recall', 'F1']
        rows = [
            self.total_gold,  # refs
            self.total_preds,  # preds
            self.correct,  # correct
            self.precision,  # precision
            self.recall,  # recall
            self.f1  # f1
        ]
        self.summary = np.array(rows, dtype=np.float32)

        self.macro_f1 = np.mean(self.f1)
        self.macro_precision = np.mean(self.precision)
        self.macro_recall = np.mean(self.recall)
        self.micro_f1 = np.sum(self.f1 * self.total_gold) / np.sum(self.total_gold)
        self.accuracy = 100 * self.confusion.diagonal().sum() / np.sum(self.total_gold)

    @classmethod
    def confusion_matrix(cls, n_classes, prediction, truth):
        matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
        assert len(prediction) == len(truth)
        for pred, gold in zip(prediction, truth):
            matrix[gold][pred] += 1
        return matrix

    def format(self, confusion=True, col_width=10, delim='\t'):
        assert col_width >= 8
        builder = []
        builder.append(["MacroF1", f"{self.macro_f1:.2f} %"])
        builder.append(["MicroF1", f"{self.micro_f1:.2f} %"])
        builder.append(["Accuracy", f"{self.accuracy:.2f} %"])
        builder.append([])
        row = ["[Class]"] + [col for col in self.col_head]
        builder.append(row)

        def number_format(n, decimals=2):
            fstr = '%d' if float(n).is_integer() else f'%.{decimals}f'
            return fstr % n

        for cls_idx, cls_name in enumerate(self.clsmap):
            row = [cls_name] + [number_format(cell) for cell in self.summary[:, cls_idx]]
            builder.append(row)

        if confusion:
            builder.append([])  # blank line
            cls_names = [cn for cn in self.clsmap]
            builder.append(["vTr Pr>"] + [c for c in cls_names] + ["[TotGold]"])
            for cls_idx, (cls_name, row) in enumerate(zip(cls_names, self.confusion)):
                row = [cls_name] + [f'{cell}' for cell in row] + [f'{self.total_gold[cls_idx]}']
                builder.append(row)

            row = ["[TotPreds]"] + [f'{cell}' for cell in self.total_preds] \
                  + [f'{self.total_gold.sum()}']
            builder.append(row)

        body = '\n'.join([delim.join(row) for row in builder])
        return body


if __name__ == '__main__':
    preds = [0, 0, 1, 1, 0, 1, 0, 1]
    truth = [0, 0, 0, 0, 1, 1, 1, 2]
    clsmap = ["cat", "dog", "goat"]
    metric = ClsMetric(prediction=preds, truth=truth, clsmap=clsmap)
    print(metric.format(delim=','))
