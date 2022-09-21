ARG CUDA_VER=11.2
ARG LINUX_TYPE=ubuntu
ARG LINUX_VER=20.04
ARG PYTHON_VER=3.8
ARG RAPIDS_VER=22.06
ARG FROM_IMAGE=rapidsai/rapidsai

FROM ${FROM_IMAGE}:${RAPIDS_VER}-cuda${CUDA_VER}-runtime-${LINUX_TYPE}${LINUX_VER}-py${PYTHON_VER}

MAINTAINER Julio Faracco <jcfaracco@gmail.com>

ENV SINGULARITYENV_TINI_SUBREAPER=1

RUN apt-key del 7fa2af80

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

COPY environment.yml /tmp/

RUN apt-get update && apt-get install git graphviz gcc python3-dev g++ -y \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/apt/lists/*

RUN conda run -n rapids pip3 install git+https://github.com/discovery-unicamp/dasf-core.git

RUN mkdir /dasf

WORKDIR /dasf

# Update jupyter environment with ipympl extension
RUN conda run -n rapids jupyter nbextension install --py --symlink --sys-prefix --overwrite ipympl
RUN conda run -n rapids jupyter nbextension enable --py --sys-prefix ipympl
