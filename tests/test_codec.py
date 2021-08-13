#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/17/21
from rtg.data.codec import NLField
import tempfile
from pathlib import Path
import shutil


class TestNLField:

    def test_train_class(self):
        args = dict(model_type="class", vocab_size=-1)
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_dir.mkdir(exist_ok=True, parents=True)
        txt_file = tmp_dir / 'classes.txt'
        model_path = tmp_dir / 'model.tsv'
        data = "\n".join("A B C D A B C D A A A A A A B B C B C A A".split())
        txt_file.write_text(data)
        field = NLField.train(**args, model_path=model_path, files=[txt_file])
        assert model_path.exists()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        assert len(field) == 4
        assert len(field) == len(field.vocab)
        assert len(field.encode_as_ids("A")) == 1
        assert field.encode_as_ids("A")[0] == 0
        assert field.encode_as_ids("B")[0] == 1
        assert field.encode_as_ids("C")[0] == 2
        assert field.encode_as_ids("D")[0] == 3
        assert field.encode_as_ids("X")[0] == -1
