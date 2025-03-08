#!/bin/bash

docker run -it --rm \
    --name "mujoco-sim" \
    --network ros-net \
    --runtime=nvidia \
    --privileged \
    -e DISPLAY=:1 \
    -e NVIDIA_VISIBLE_DEVICES=all\
    -e NVIDIA_DRIVER_CAPABILITIES=all\
    --gpus all \
    -v /dev:/dev \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    CONTAINER_REGISTRY/mujoco-insertion:v0.4-gpu \
    $*
