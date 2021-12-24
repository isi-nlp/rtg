
from io import StringIO

def sanity_check_experiment(exp, samples=True, shared_vocab=True):
    """
    Bunch of sanity checks on experiment dir
    :param exp:
    :return:
    """
    assert exp._config_file.exists()
    assert exp.data_dir.exists()
    assert exp.train_db.exists()
    assert exp.train_db.stat().st_size > 0
    assert exp.valid_file.exists()
    if samples:
        assert exp.samples_file.exists()
    if shared_vocab:
        assert exp._shared_field_file.exists()
    else:
        assert exp._src_field_file.exists()
        assert exp._tgt_field_file.exists()
    assert exp._prepared_flag.exists()
    assert exp._trained_flag.exists()
    assert len(list(exp.model_dir.glob('*.pkl'))) > 0
    assert len((exp.model_dir / 'scores.tsv').read_text().splitlines()) > 0


def run_decode(exp_dir, sentences):
    assert isinstance(sentences, list)
    from rtg.decode import main as decode_cli
    buffer = StringIO()
    decode_cli(exp_dir=exp_dir, input=[sentences], output=[buffer], skip_check=True, max_src_len=200)
    lines = buffer.getvalue().splitlines()
    buffer.close()
    return lines