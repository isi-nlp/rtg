# What is this: USC ISI Coral team's MT pipeline
# Authors:
#      - Thamme Gowda
# Created : Oct 20, 2020

FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Install curl, python3.7 and pip
RUN apt-get update && apt-get install -y \
    curl \
    python3.7 \
    python3-pip \
    python3.7-dev \
    build-essential \
    git \
    && apt-get autoremove --purge

# Update pip
RUN python3.7 -m pip install --upgrade pip  && ln -s /usr/bin/python3.7 /usr/bin/python

#Make non-root user;
RUN useradd --create-home rtguser
#RUN chown -Rv rtguser:rtguser /home/rtguser

WORKDIR /home/rtguser
USER rtguser

# pip installed bins go here, they needs to be in PATH
RUN mkdir -p /home/rtguser/.local/bin /home/rtguser/rtg
ENV CUDA_HOME="/usr/local/cuda-10.2/"
ENV PATH="/home/rtguser/.local/bin:/usr/local/cuda-10.2/bin:${PATH}"

#COPY --chown=rtguser:rtguser . /home/rtguser/rtg/
#   && cd /home/rtguser/rtg && pip install --editable . \

RUN pip install --user torch==1.6 flask==1.1.2 uwsgi rtg==0.5.0  \
   && pip cache purge

CMD bash
