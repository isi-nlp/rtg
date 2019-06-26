import setuptools

long_description="""
Reader-Translator-Generator (RTG) is a Neural Machine Translation toolkit based on pytorch.

"""


setuptools.setup(
     name='rtg',
     version='0.1',
     scripts=['bin/rtg'] ,
     author="Thamme Gowda",
     author_email="tg@isi.edu",
     description="Reader Translator Generator ( RTG , NMT ) ",
     long_description=long_description,
   long_description_content_type="text/plain",
     url="https://github.com/isi-nlp/rtg",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: Apache License",
         "Operating System :: OS Independent",
     ],
 )
