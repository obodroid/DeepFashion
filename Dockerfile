FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    nano \
    git \
    wget \
    net-tools \
    vlc \
    vim \
    zip \
    unzip

# Tensorflow (optional)
RUN sudo apt-get install -y python-pip python-dev python-virtualenv

# for Python 2.7
RUN virtualenv --system-site-packages tensorflow121_py27_gpu

COPY requirements.txt /tmp/requirements.txt

# # for Python 2.7 and GPU
RUN /bin/bash -c "source tensorflow121_py27_gpu/bin/activate \
    && pip install --upgrade tensorflow-gpu \
    && sudo apt install -y python-tk \
    && pip install -r /tmp/requirements.txt"