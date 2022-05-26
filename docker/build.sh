#!/bin/bash

docker build --rm  \
    --no-cache \
    --build-arg USERNAME=${USER} \
    --build-arg USER_ID=$(id -u ${USER}) \
    --build-arg GROUP_ID=$(id -g ${USER}) \
    --tag ${USER}/upper_body:latest .
