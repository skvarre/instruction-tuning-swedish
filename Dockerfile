FROM continuumio/miniconda3

WORKDIR /app 

COPY . .
ENV CONDA_DEFAULT_TIMEOUT=100
RUN conda env create -f environment.yml --quiet --yes

EXPOSE 22

ENV PATH /opt/conda/envs/pytorch-env/bin:$PATH

RUN which python