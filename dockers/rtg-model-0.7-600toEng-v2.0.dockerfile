# What is this: USC ISI Coral team's MT pipeline
# Authors:
#      - Thamme Gowda
# Created : March 15, 2022

FROM tgowda/rtg:0.7-ub20.04-py3.9_tr1.10_cu11.4
# Download pretrained model from google drive

RUN cd /home/rtguser/ && \
   curl -O http://rtg.isi.edu/many-eng/models/many-eng-v2.0-rtg600eng1024d_ful74k.tgz && \
   tar xvf many-eng-v2.0-rtg600eng1024d_ful74k.tgz  --one-top-level=many-eng-v2.0-600toeng --strip-components 1 && \
   rm *.tgz

#CMD rtg-serve  /home/rtguser/
#CMD python -m rtg.serve /home/rtguser/many-eng-v2.0-600toeng
CMD uwsgi --http 0.0.0.0:6060 --module rtg.serve.app:app --pyargv "/home/rtguser/many-eng-v2.0-600toeng"

