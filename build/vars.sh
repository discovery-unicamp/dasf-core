#!/bin/bash

function pushd() {
    command pushd "$@" > /dev/null
}

function popd() {
    command popd "$@" > /dev/null
}

function GET_CONTAINER_CMD() {
    OLDIFS="$IFS"
    IFS=":"

    for P in $PATH; do
        if test -f "$P/podman"; then
            CONTAINER_CMD=podman
            break
        else
            CONTAINER_CMD=docker
        fi
    done

    IFS=$OLDIFS

    echo $CONTAINER_CMD
}

DOCKERFILE_DIR=docker/
WORKDIR=$(realpath $(pwd))
IMAGE=dasf
PORT=8891
