import tempfile
import shutil

from rtg import load_conf, TranslationExperiment
from rtg.cli.pipeline import Pipeline
from . import sanity_check_experiment


def test_macro_cross_entropy():
    tmp_dir = tempfile.mkdtemp()
    config = load_conf('experiments/loss/tfmnmt.macro_kldiv.yml')
    exp = TranslationExperiment(tmp_dir, config=config, read_only=False)
    exp.config['trainer'].update(dict(steps=50, check_point=25))
    exp.config['prep']['num_samples'] = 0
    Pipeline(exp).run(run_tests=False)
    sanity_check_experiment(exp)
    print(f"Cleaning up {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
