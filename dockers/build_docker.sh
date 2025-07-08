#!/bin/bash

FILE=$1
IMAGE_NAME="bench:latest"

echo "Building Docker image: ${FILE}"

docker build -t ${IMAGE_NAME} -f ${FILE} .

echo "Done - Docker image name: ${IMAGE_NAME}"
