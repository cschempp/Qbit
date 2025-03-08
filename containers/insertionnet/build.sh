#!/bin/bash -e
BASEDIR=$(dirname "$0")
ROS_DISTRO="humble"

echo "Using Dockerfile: $BASEDIR"
echo "From Context: $PWD"

# # build image
DOCKER_BUILDKIT=1 docker build \
  $@ \
  -t CONTAINER_REGISTRY/insertionnet:peg-v0.3 \
  -f $BASEDIR/Dockerfile \
  $PWD
