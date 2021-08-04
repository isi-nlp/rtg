# What is this: USC ISI Coral team's MT pipeline
# Authors:
#      - Thamme Gowda
# Created : Oct 20, 2020

FROM tgowda/rtg:v0.5.0-torch1.6

# Download pretrained model from google drive

# gdown https://drive.google.com/uc?id=14j1DPHHjnW27ixFnMyICa9MdjQH0_n9s -O rtgv0.5-768d9L6L-512K64K-datav1.tgz && \

RUN cd /home/rtguser/ && \
   curl -O http://rtg.isi.edu/many-eng/models/rtg500eng-tfm9L6L768d-bsz720k-stp200k-ens05.tgz && \
   tar xvf rtg500eng-tfm9L6L768d-bsz720k-stp200k-ens05.tgz --one-top-level=rtg500eng-tfm9L6L768d-bsz720k-stp200k-ens05 --strip-components 1 && \
   rm rtg500eng-tfm9L6L768d-bsz720k-stp200k-ens05.tgz

#RUN pip install sacremoses

#CMD rtg-serve  /home/rtguser/rtg002-768d-9L6L-512k64k-datav1
#CMD python -m rtg.serve /home/rtguser/rtg002-768d-9L6L-512k64k-datav1 
CMD uwsgi --http 0.0.0.0:6060 --module rtg.serve.app:app --pyargv "/home/rtguser/rtg500eng-tfm9L6L768d-bsz720k-stp200k-ens05"

