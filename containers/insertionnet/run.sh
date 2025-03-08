#!/bin/bash

docker run -it --rm \
    --name "insertion-net" \
    --network ros-net \
    --runtime=nvidia \
    -p 50053:50052 \
    --gpus all \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    CONTAINER_REGISTRY/insertionnet:peg-v0.1 \
    $*


