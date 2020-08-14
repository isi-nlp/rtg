ARG experiment_dir
FROM python:3.7
# RUN echo ${experiment_dir}
# RUN echo "hello world"
# COPY ${experiment_dir} /experiment/
COPY ./experiments/sample-exp/ /experiment/
COPY . /rtg
WORKDIR /rtg
RUN pip install -e ./
RUN python -m rtg.pipeline /experiment/
CMD python -m rtg.deploy /experiment/