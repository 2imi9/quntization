#! /bin/bash

IMAGE_NAME="bench:latest"

# Get Project Root
CURRENT_DIR="${PWD##*/}"
if [[ "$CURRENT_DIR" == *dockers* ]]; then
    PROJECT_DIR=$(dirname ${PWD})
else
    PROJECT_DIR=${PWD}
fi
echo "Project directory: ${PROJECT_DIR}"

# Get Data Dir
DATA_DIR="${HOME}/bench_data/"

# Launch
docker run -it --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ${DATA_DIR}:/data \
    -v ${PROJECT_DIR}:/app \
    --name jiaqi_bench_dev \
    -p 8000:8000 \
    --ipc=host \
    ${IMAGE_NAME} \
    bash
