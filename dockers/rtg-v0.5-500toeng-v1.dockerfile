# What is this: USC ISI Coral team's MT pipeline
# Authors:
#      - Thamme Gowda
# Created : Oct 20, 2020

FROM tgowda/rtg-torch1.6-cuda10.2:0.5 

# Download pretrained model from google drive

RUN cd /home/rtguser/ && \
   gdown https://drive.google.com/uc?id=14j1DPHHjnW27ixFnMyICa9MdjQH0_n9s -O rtgv0.5-768d9L6L-512K64K-datav1.tgz && \
   tar xvf rtgv0.5-768d9L6L-512K64K-datav1.tgz --one-top-level=rtgv0.5-768d9L6L-512K64K-datav1 --strip-components 1 && \
   rm rtgv0.5-768d9L6L-512K64K-datav1.tgz

RUN pip install sacremoses

#CMD rtg-serve  /home/rtguser/rtgv0.5-768d9L6L-512K64K-datav1
#CMD python -m rtg.serve /home/rtguser/rtgv0.5-768d9L6L-512K64K-datav1
CMD uwsgi --main --http 0.0.0.0:6060 --module rtg.serve.app:app --pyargv "/home/rtguser/rtgv0.5-768d9L6L-512K64K-datav1"
