#!/bin/bash

xhost +local:docker
DATA_DIR=${1:-$(pwd)/data}

echo "Starting MSALT container..."
echo "Mapping Data Directory: $DATA_DIR"

docker run -it --rm \
    --gpus all \
        --net=host \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$DATA_DIR:/app/data" \
        --volume="$(pwd)/annotations:/app/annotations" \
        --name salt_app \
        salt_tool

xhost -local:docker