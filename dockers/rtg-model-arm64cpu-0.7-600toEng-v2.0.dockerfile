#  RTG docker, without cuda, for arm64 cpu such as Apple silicon, Qualcomm snapdragon, Microsoft SQ etc
# Authors:
#      - Thamme Gowda
# Created : March 15, 2022
FROM ubuntu:22.04

# suppress prompts https://stackoverflow.com/a/67452950/1506477
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
 && apt install -y curl python3.9 python3-pip python3.9-dev \
   build-essential git locales locales-all \
 && apt-get autoremove --purge

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Update pip
RUN  ln -s /usr/bin/python3.9 /usr/bin/python && python -m pip install --upgrade pip

#Make non-root user;
RUN useradd --create-home rtguser
#RUN chown -Rv rtguser:rtguser /home/rtguser

WORKDIR /home/rtguser
USER rtguser

# pip installed bins go here, they needs to be in PATH
RUN mkdir -p /home/rtguser/.local/bin /home/rtguser/rtg
ENV PATH="/home/rtguser/.local/bin:${PATH}"

#COPY --chown=rtguser:rtguser . /home/rtguser/rtg/
#   && cd /home/rtguser/rtg && pip install --editable . \

RUN pip install --user torch==1.10.2 flask==2.0.3 uwsgi rtg==0.7 setuptools==59.5.0  && pip cache purge

CMD bash

RUN cd /home/rtguser/ && \
   curl -O http://rtg.isi.edu/many-eng/models/many-eng-v2.0-rtg600eng1024d_ful74k.tgz && \
   tar xvf many-eng-v2.0-rtg600eng1024d_ful74k.tgz  --one-top-level=many-eng-v2.0-600toeng --strip-components 1 && \
   rm *.tgz

#CMD rtg-serve  /home/rtguser/
#CMD python -m rtg.serve /home/rtguser/many-eng-v2.0-600toeng
CMD uwsgi --http 0.0.0.0:6060 --module rtg.serve.app:app --pyargv "/home/rtguser/many-eng-v2.0-600toeng"

