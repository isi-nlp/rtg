


def sanity_check_experiment(exp):
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
    assert exp.samples_file.exists()
    assert exp._shared_field_file.exists()
    assert exp._prepared_flag.exists()
    assert exp._trained_flag.exists()
    assert len(list(exp.model_dir.glob('*.pkl'))) > 0
    assert len((exp.model_dir / 'scores.tsv').read_text().splitlines()) > 0
