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
assert __version__, 'Could not find __version__ in __init__.py'

setuptools.setup(
    name='rtg',
    version=__version__,
    author="Thamme Gowda",
    author_email="tg@isi.edu",
    description="Reader Translator Generator(RTG), a Neural Machine Translator(NMT) toolkit based on Pytorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://isi-nlp.github.io/rtg/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=['any'],
    install_requires=[
        'ruamel.yaml >= 0.17.10',
        'sacrebleu == 1.4.14',
        'sentencepiece >= 0.1.85',
        'tensorboard >= 2.6.0',
        'tqdm >= 4.45.0',
        'mosestokenizer >= 1.0.0',
        'nlcodec >= 0.4.0',
        'torch >= 1.8.0',
        'sacremoses >= 0.0.45',
        'portalocker >= 2.0.0',
        'torchtext >= 0.10.0',
        'pyspark >= 3.0.0'
    ],
    extra_requires={
        'big': ['pyspark >= 3.0'],
        'extras': ['scipy >= 1.4'],
        'serve': ['flask >= 1.1.2'],
    },
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
            'rtg-serve=rtg.serve.app:main',
            'rtg-syscomb=rtg.syscomb.__main__:main',
            'rtg-launch=rtg.distrib.launch:main',
            'rtg-params=rtg.tool.params:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
