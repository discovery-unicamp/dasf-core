source vars.sh

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

EXTRA_ARGS=$@

$CONTAINER_CMD run --gpus all -it --rm  -p $PORT:$PORT -e SHELL="/bin/bash" --network=host $EXTRA_ARGS $IMAGE python -m jupyterlab --allow-root --ServerApp.port $PORT --no-browser --ServerApp.ip='0.0.0.0'
