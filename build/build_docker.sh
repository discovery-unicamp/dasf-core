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

CONTAINER_CMD=$(GET_CONTAINER_CMD)

DOCKERFILE_DIR=$DOCKERFILE_DIR/$ARCH_TYPE

# Copy respective conda settings
cp conda/*.yml $DOCKERFILE_DIR
cp scripts/.bashrc $DOCKERFILE_DIR
cp scripts/entrypoint.sh $DOCKERFILE_DIR

pushd $DOCKERFILE_DIR
$CONTAINER_CMD build -t $IMAGE . 

# Clean up
echo "Cleaning up..."
popd
rm -rf $DOCKERFILE_DIR/*.yml $DOCKERFILE_DIR/.bashrc $DOCKERFILE_DIR/entrypoint.sh
