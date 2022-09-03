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

RUN apt-get update && apt-get install git wget bzip2 graphviz gcc python3-dev g++ -y

RUN conda create --clone rapids --name dasf

RUN conda env update -n dasf --file /tmp/environment.yml

RUN conda init --system -q bash \
    && grep -v '[ -z "\$PS1" ] && return' /root/.bashrc  > /opt/conda/bashrc   # this line has been modified

RUN apt-get clean autoremove --yes \
    && rm -rf /var/lib/apt/lists/*

COPY .bashrc /root/

COPY entrypoint.sh /usr/local/bin/

RUN mkdir /dasf

WORKDIR /dasf

ENV CONDA_PREFIX /opt/conda/envs/dasf
ENV CONDA_PROMPT_MODIFIER (dasf)
ENV CONDA_DEFAULT_ENV dasf

# Update jupyter environment with ipympl extension
RUN conda run -n dasf jupyter nbextension install --py --symlink --sys-prefix --overwrite ipympl
RUN conda run -n dasf jupyter nbextension enable --py --sys-prefix ipympl

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

