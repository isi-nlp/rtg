
# Authors:
#      - Thamme Gowda
# Created : Oct 19, 2022

# parent
FROM tgowda/rtg:0.7.1-ub20.04-py3.9_tr1.10_cu11.4

# Download pretrained model from rtg demo website
RUN cd /home/rtguser/ && \
   curl -O http://rtg.isi.edu/many-eng/models/many-eng-v2.0-rtg600eng1024d_ful74k.tgz && \
   tar xvf many-eng-v2.0-rtg600eng1024d_ful74k.tgz  --one-top-level=many-eng-v2.0-600toeng --strip-components 1 && \
   rm *.tgz

# install/upgrade the desired versions of libs
RUN pip install --user torch==1.12 flask==2.1 uwsgi rtg==0.7.2 \
 && pip cache purge

#CMD rtg-serve  /home/rtguser/
#CMD python -m rtg.serve /home/rtguser/many-eng-v2.0-600toeng
CMD uwsgi --http 0.0.0.0:6060 --module rtg.serve.app:app --pyargv "/home/rtguser/many-eng-v2.0-600toeng"
