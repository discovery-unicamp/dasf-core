#!/bin/bash

PORT=8891
TYPE="gpu"
FORMAT="docker"

function print_help() {
    echo "Usage: ./build_docker.sh [ARCH_TYPE]"
    echo ""
    echo "Standard options:"
    echo "    -h or -help                      Display this help and exit"
    echo "    --device ARCH_TYPE:              Build a container based on arch type."
    echo "                                     Use 'gpu' for GPU container-based architecture."
    echo "                                     Use 'cpu' for CPU container-based architecture (default='$TYPE')."
    echo "    --port PORT                      Use the prefered port to start the server (default='$PORT')."
    echo "    --format FORMAT                  Select the container backend for this service."
    echo "                                     Use 'docker' for Docker images."
    echo "                                     Use 'singularity' for SIF images (default='$FORMAT')."
    echo ""
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_help
      exit 0
      ;;
    --device)
      TYPE="$2"
      shift
      shift
      ;;
    --format)
      FORMAT="$2"
      shift
      shift
      ;;
    --port|-p)
      PORT="$2"
      shift
      shift
      ;;
    *)
      break;
  esac
done

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

[[ -z "${CONTAINER_CMD}" ]] && CONTAINER_CMD=$(GET_CONTAINER_CMD)

EXTRA_ARGS=$@

if [[ "$FORMAT" == "docker" ]]; then
    if [[ "$TYPE" == "gpu" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --gpus all"
    fi

    $CONTAINER_CMD run -it --rm  -p $PORT:$PORT -e SHELL="/bin/bash" --network=host $EXTRA_ARGS dasf:$TYPE python3 -m jupyterlab --allow-root --ServerApp.port $PORT --no-browser --ServerApp.ip='0.0.0.0'
elif [[ "$FORMAT" == "singularity" ]]; then
    if [[ "$TYPE" == "gpu" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --nv"
    fi

    singularity exec $EXTRA_ARGS dasf_$TYPE.sif python3 -m jupyterlab --allow-root --ServerApp.port $PORT --no-browser --ServerApp.ip=0.0.0.0
fi
