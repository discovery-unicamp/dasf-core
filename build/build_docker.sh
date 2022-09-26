source vars.sh

function print_help() {
    echo "Usage: ./build_docker.sh [ARCH_TYPE]"
    echo ""
    echo "Standard options:"
    echo "    -h or -help     Display this help and exit"
    echo "    ARCH_TYPE:      Build a container based on arch type."
    echo "                    Use 'gpu' for GPU container-based architecture (default)."
    echo "                    Use 'cpu' for CPU container-based architecture."
    echo ""
}

# Default architecture
ARCH_TYPE="gpu"

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_help
      exit 0
      ;;
    cpu)
      ARCH_TYPE="cpu"
      shift
      ;;
    gpu)
      ARCH_TYPE="gpu"
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

DEVICE_TARGET=$ARCH_TYPE

function FIND_CMD() {
    if ! command -v $1 &> /dev/null
    then
        echo $2
        exit -1
    fi
}

FIND_CMD hpccm "Binary 'hpccm' could not be found: install HPC container maker first."

mkdir -p $DOCKERFILE_DIR

hpccm --recipe hpccm/build_docker.py \
      --userarg device-target=$DEVICE_TARGET \
                devel=$IS_DEVEL \
                rapids-version=$RAPIDS_VERSION \
                cuda-version=$CUDA_VERSION \
                ubuntu-version=$UBUNTU_VERSION \
      --format $FORMAT > $DOCKERFILE_DIR/$OUTPUT_FILE

if [[ "$FORMAT" == "docker" ]]; then
    FIND_CMD docker "Docker binaries are not found."
    docker build $DOCKERFILE_DIR
else
    FIND_CMD singularity "Singularity binaries are not found."
    singularity build dasf.sif $DOCKERFILE_DIR/$OUTPUT_FILE
fi

rm -rf $DOCKERFILE_DIR
