import tempfile
import shutil

from rtg import Pipeline, load_conf, MODELS


def test_hf_comet():
    tmp_dir = tempfile.mkdtemp()
    try:
        config = load_conf('tests/experiments/hf-comet-classifier.conf.yml')
        model_type = config['model_type']
        exp = MODELS[model_type].Experiment(tmp_dir, config=config, read_only=False)

        exp.config['model_args'].update(dict(freeze_encoder=True))
        exp.config['trainer'].update(dict(steps=50, check_point=25, batch_size=[4000, 4]))
        # use smaller test set as validation -- for fast validations
        if exp.model_dir.exists():
            shutil.rmtree(exp.model_dir)
            exp.model_dir.mkdir()

        pipe = Pipeline(exp)
        pipe.run(run_tests=False)
        assert len(list(exp.model_dir.glob('*.pkl'))) > 0
    finally:
        print(f"Cleaning up {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rtg_comet():
    tmp_dir = tempfile.mkdtemp()
    try:
        config = load_conf('tests/experiments/rtg-comet-classifier.conf.yml')
        model_type = config['model_type']
        exp = MODELS[model_type].Experiment(tmp_dir, config=config, read_only=False)
        exp.config['model_args'].update(dict(freeze_encoder=False))
        # exp.config['trainer'].update(dict(steps=5000, check_point=500, batch_size=[4000, 8]))
        exp.config['trainer'].update(dict(steps=100, check_point=50, batch_size=[4000, 8]))
        if exp.model_dir.exists():
            shutil.rmtree(exp.model_dir)
            exp.model_dir.mkdir()

        pipe = Pipeline(exp)
        pipe.run(run_tests=False)
        assert len(list(exp.model_dir.glob('*.pkl'))) > 0
    finally:
        print(f"Cleaning up {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
