FROM tensorflow/tensorflow:1.15.2-gpu
WORKDIR /wasr
RUN set -euo pipefail && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends python-opencv && \
    pip3 install opencv-python scipy pillow && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
