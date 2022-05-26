FROM nvidia/cuda:10.1-cudnn7-devel

# Add UTF-8 support
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8
ENV PYTHONIOENCODING='UTF-8' LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8' PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        bash \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        make \
        protobuf-compiler \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-scipy  \
        python3-setuptools \
        wget \
        tmux \
        htop \
        vim
RUN pip3 install --upgrade pip
RUN pip3 install scikit-build scikit-learn
RUN pip3 install matplotlib tqdm Pillow opencv-python==4.5.5.64 colored pyyaml seaborn
RUN pip3 install mxnet-cu101mkl
RUN pip3 install gluoncv
RUN pip3 install wandb
RUN pip3 install torch torchvision pytorch-forecasting
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
RUN pip3 install albumentations
RUN pip3 install hydra-core --upgrade
RUN pip install timm
WORKDIR /workspace

RUN rm -rf /var/lib/apt/lists/*
