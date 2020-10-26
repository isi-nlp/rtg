# What is this: USC ISI Coral team's MT pipeline
# Authors:
#      - Thamme Gowda
# Created : Oct 20, 2020

FROM tgowda/rtg:v0.5.0-torch1.6

# Download pretrained model from google drive

# gdown https://drive.google.com/uc?id=14j1DPHHjnW27ixFnMyICa9MdjQH0_n9s -O rtgv0.5-768d9L6L-512K64K-datav1.tgz && \

RUN cd /home/rtguser/ && \
   gdown https://drive.google.com/uc?id=1Oci6fM3lr0KcHb6rgINcG4EbnzN0uFws -O rtg002-768d-9L6L-512k64k-datav1.tgz  && \   
   tar xvf rtg002-768d-9L6L-512k64k-datav1.tgz --one-top-level=rtg002-768d-9L6L-512k64k-datav1 --strip-components 1 && \
   rm rtg002-768d-9L6L-512k64k-datav1.tgz 

#RUN pip install sacremoses

#CMD rtg-serve  /home/rtguser/rtg002-768d-9L6L-512k64k-datav1
#CMD python -m rtg.serve /home/rtguser/rtg002-768d-9L6L-512k64k-datav1 
CMD uwsgi --http 0.0.0.0:6060 --module rtg.serve.app:app --pyargv "/home/rtguser/rtg002-768d-9L6L-512k64k-datav1"

