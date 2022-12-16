#!/bin/bash

# Default variables

ARCH_TYPE="gpu"
FORMAT="docker"
OUTPUT_FILE="Dockerfile"
IS_DEVEL="False"
RAPIDS_VERSION="22.08"
CUDA_VERSION="11.2"
UBUNTU_VERSION="20.04"
PYTHON_VERSION="3.9"
DOCKERFILE_DIR=docker/

# For local hpccm installs
OLD_PATH=$PATH
export PATH=$PATH:$HOME/.local/bin/

function print_help() {
    echo "Usage: ./build_docker.sh [ARCH_TYPE]"
    echo ""
    echo "Standard options:"
    echo "    -h or -help                      Display this help and exit"
    echo "    --device ARCH_TYPE:              Build a container based on arch type."
    echo "                                     Use 'gpu' for GPU container-based architecture."
    echo "                                     Use 'cpu' for CPU container-based architecture (default='$ARCH_TYPE')."
    echo "    --rapids-version RAPIDS_VERSION  Defines which version of RAPIDS AI will be used (default='$RAPIDS_VERSION')."
    echo "    --cuda-version CUDA_VERSION      Defines which version of CUDA will be used (default='$CUDA_VERSION')."
    echo "    --os-version OS_VERSION          Defines which version of the container will be used (default='$UBUNTU_VERSION')."
    echo "    --python-version PYTHON_VERSION  Defines which version of the python interpreter will be used (default='$PYTHON_VERSION')."
    echo "    --format FORMAT                  Select the container backend for this build."
    echo "                                     Use 'docker' for Docker images."
    echo "                                     Use 'singularity' for SIF images (default='$FORMAT')."
    echo ""
}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_help
      exit 0
      ;;
    --device)
      ARCH_TYPE="$2"
      shift
      shift
      ;;
    --rapids-version)
      RAPIDS_VERSION="$2"
      shift
      shift
      ;;
    --cuda-version)
      CUDA_VERSION="$2"
      shift
      shift
      ;;
    --os-version)
      UBUNTU_VERSION="$2"
      shift
      shift
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift
      shift
      ;;
    --format)
      FORMAT="$2"
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [[ "${#POSITIONAL_ARGS[@]}" -gt 1 ]]; then
  echo "Invalid number of positional arguments"
  exit 1
elif [[ "${#POSITIONAL_ARGS[@]}" -eq 1 ]]; then
  ARCH_TYPE="${POSITIONAL_ARGS[0]}"
fi

if [[ "$ARCH_TYPE" != "cpu" && "$ARCH_TYPE" != "gpu" ]]; then
    echo "Invalid '--device' type. Check -h|--help for further details."
    exit 1
fi

if [[ "$FORMAT" != "docker" && "$FORMAT" != "singularity" ]]; then
    echo "Invalid container backend for '--format'. Check -h|--help for further details."
    exit 1
fi

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

CONTAINER_CMD=$(GET_CONTAINER_CMD)

function FIND_CMD() {
    if ! command -v $1 &> /dev/null
    then
        echo $2
        exit -1
    fi
}

FIND_CMD hpccm "Binary 'hpccm' could not be found: install HPC container maker first.
    Check https://github.com/NVIDIA/hpc-container-maker for more details"

mkdir -p $DOCKERFILE_DIR

hpccm --recipe hpccm/build_docker.py \
      --userarg device-target=$ARCH_TYPE \
                devel=$IS_DEVEL \
                rapids-version=$RAPIDS_VERSION \
                cuda-version=$CUDA_VERSION \
                ubuntu-version=$UBUNTU_VERSION \
                python-version=$PYTHON_VERSION \
      --format $FORMAT > $DOCKERFILE_DIR/$OUTPUT_FILE

if [[ "$FORMAT" == "docker" ]]; then
    FIND_CMD $CONTAINER_CMD "Docker binaries are not found."
    $CONTAINER_CMD build $DOCKERFILE_DIR -t dasf:$ARCH_TYPE
else
    FIND_CMD singularity "Singularity binaries are not found."
    singularity build dasf_$ARCH_TYPE.sif $DOCKERFILE_DIR/$OUTPUT_FILE
fi

rm -rf $DOCKERFILE_DIR

export PATH=$OLD_PATH
