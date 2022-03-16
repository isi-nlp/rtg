# What is this: USC ISI Coral team's MT pipeline
# Authors:
#      - Thamme Gowda
# Created : Oct 20, 2020

#FROM nvidia/cuda:10.2-devel-ubuntu18.04
#FROM nvidia/cuda:11.1-devel-ubuntu20.04
FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04

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
ENV CUDA_HOME="/usr/local/cuda/"
ENV PATH="/home/rtguser/.local/bin:/usr/local/cuda/bin:${PATH}"

#COPY --chown=rtguser:rtguser . /home/rtguser/rtg/
#   && cd /home/rtguser/rtg && pip install --editable . \

RUN pip install --user torch==1.10.2 flask==2.0.3 uwsgi rtg==0.7  \
   && pip cache purge

CMD bash
