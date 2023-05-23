from rtg.comet import HfTransformerExperiment as Experiment
from pathlib import Path
from rtg import Pipeline
import shutil


def test_comet_classifier():
    exp = Experiment('tests/experiments/comet-cls', read_only=True)
    exp.config['model_args'].update(dict(freeze_encoder=True))
    exp.config['trainer'].update(dict(steps=50, check_point=25, batch_size=[4000, 4]))
    # use smaller test set as validation -- for fast validations
    exp.config['prep'].update(dict(valid_src="${data}.test.texts", valid_tgt="${data}.test.label"))

    if exp.model_dir.exists():
        shutil.rmtree(exp.model_dir)

    pipe = Pipeline(exp)
    pipe.run(run_tests=False)
