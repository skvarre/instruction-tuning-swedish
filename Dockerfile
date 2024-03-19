FROM ubuntu:latest

WORKDIR /app 

# Install dependencies
RUN apt-get update && apt-get install -y \
    openssh-server \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /miniconda \
    && rm miniconda.sh

ENV PATH /miniconda/bin:$PATH

COPY . .
ENV CONDA_DEFAULT_TIMEOUT=100

RUN pip install --upgrade pip
RUN bash init-ssh.sh tim tim 

RUN conda env create -f environment.yml --quiet --yes
ENV PATH /opt/conda/envs/pytorch-env/bin:$PATH

EXPOSE 22

RUN which python
