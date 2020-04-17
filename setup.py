import setuptools
from pathlib import Path
import re

long_description = Path('README.md').read_text(encoding='utf-8', errors='ignore')


vpat = re.compile(r"""__version__\s*=\s*['"]([^'"]*)['"]""")
__version__ = None
for line in Path('rtg/__init__.py').read_text().splitlines():
    line = line.strip()
    if vpat.match(line):
        __version__ = vpat.match(line)[1]

print(f"Going to install rtg {__version__}")
assert __version__, 'Coulnt find __version__ in __init__.py'

setuptools.setup(
    name='rtg',
    version=__version__,
    scripts=['bin/rtg'],
    author="Thamme Gowda",
    author_email="tg@isi.edu",
    description="Reader Translator Generator ( RTG), a Neural Machine Translator toolkit ",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/isi-nlp/rtg",
    packages=['rtg'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'ruamel.yaml >= 0.16.10',
        'sacrebleu == 1.4.6',
        'scipy == 1.2.1',
        'sentencepiece == 0.1.85',
        'tensorboard == 2.2.1',
        'tqdm == 4.45.0',
        'mosestokenizer >= 1.0.0',
        'nlcodec >= 0.2.0',
        'torch == 1.4'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'rtg-pipe=rtg.pipeline:main',
            'rtg-decode=rtg.decode:main',
            'rtg-decode-pro=rtg.decode_pro:main',
            'rtg-export=rtg.export:main',
            'rtg-prep=rtg.prep:main',
            'rtg-train=rtg.train:main',
            'rtg-fork=rtg.fork:main',
            'rtg-syscomb=rtg.syscomb.__main__:main',
        ],
    }
)
