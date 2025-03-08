#!/bin/bash -e
BASEDIR=$(dirname "$0")

echo "Using Dockerfile: $BASEDIR"
echo "From Context: $PWD"

# # build image
DOCKER_BUILDKIT=1 docker build \
  $@ \
  -t CONTAINER_REGISTRY/mujoco-insertion:v0.4-gpu \
  -f $BASEDIR/Dockerfile \
  $PWD
