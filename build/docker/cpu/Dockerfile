ARG LINUX_TYPE=ubuntu
ARG LINUX_VER=20.04

FROM ${LINUX_TYPE}:${LINUX_VER}

MAINTAINER Julio Faracco <jcfaracco@gmail.com>

ENV SINGULARITYENV_TINI_SUBREAPER=1

COPY environment.yml /tmp/

# Set timezone to Greenwich
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/London" apt-get install git wget bzip2 graphviz -y \
    && wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1 \
    && cp micromamba /usr/bin/ \
    && touch /root/.bashrc

RUN micromamba create -f /tmp/environment.yml -y

RUN micromamba shell init -s bash

RUN apt-get clean autoremove --yes \
    && rm -rf /var/lib/apt/lists/*

COPY .bashrc /root/

COPY entrypoint.sh /usr/local/bin/

RUN mkdir /dasf

WORKDIR /dasf

ENV CONDA_PREFIX /root/micromamba/envs/dasf
ENV CONDA_PROMPT_MODIFIER (dasf)
ENV CONDA_DEFAULT_ENV dasf

# Update jupyter environment with ipympl extension
RUN micromamba run -n dasf jupyter nbextension install --py --symlink --sys-prefix --overwrite ipympl
RUN micromamba run -n dasf jupyter nbextension enable --py --sys-prefix ipympl

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
