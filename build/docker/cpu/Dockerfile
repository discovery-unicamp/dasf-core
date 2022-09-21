ARG LINUX_TYPE=ubuntu
ARG LINUX_VER=20.04

FROM ${LINUX_TYPE}:${LINUX_VER}

MAINTAINER Julio Faracco <jcfaracco@gmail.com>

ENV SINGULARITYENV_TINI_SUBREAPER=1

# Set timezone to Greenwich
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/London" apt-get install git graphviz gcc g++ python3-pip python3-dev -y \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install git+https://github.com/discovery-unicamp/dasf-core.git jupyterlab

RUN mkdir /dasf

WORKDIR /dasf

# Update jupyter environment with ipympl extension
RUN jupyter nbextension install --py --symlink --sys-prefix --overwrite ipympl
RUN jupyter nbextension enable --py --sys-prefix ipympl
