#!/bin/bash
docker run -it \
    --shm-size 16G \
    --log-driver=none \
    --gpus device=0 \
    --volume=${PWD}/../../Attributes-pytorch/data/Records:/workspace/data \
    --volume=${PWD}/..:/workspace/code \
    --entrypoint /bin/bash \
   ${USER}/upper_body:latest
