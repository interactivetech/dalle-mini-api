# Flask app Dockerfile
# FROM ubuntu:18.04
#FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
#FROM tensorflow/tensorflow:2.4.0-gpu
FROM gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010
#FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# declare the image name
#ENV IMG_NAME=11.6.0-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    #JAXLIB_VERSION=0.3.0

# install python3-pip
#RUN apt update && apt install python3-pip -y

# install dependencies via pip
#RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
ENV DEBIAN_FRONTEND=noninteractive
#RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list

# Install needed dependencies for our app to run.
RUN apt-get update -y \
  && apt-get install -y \
    gcc \
    libffi-dev \
    musl-dev \
    git \
    sudo \
    python3-pip \
    python3-dev \
    && cd /usr/local/bin \
    && ln -sf /usr/bin/python3 python \
    && pip3 --no-cache-dir install --upgrade pip 
#  RUN apt install -y g++ jpeg-dev zlib-dev libjpeg make
# Remove apt cache to make the image smaller
RUN rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel
# Change the CUDA version if it doesn't match the installed version.
#ARG JAX_CUDA_VERSION=10.0
#COPY install_cuda.sh /install_cuda.sh
#RUN chmod +x /install_cuda.sh
#RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
#  /install_cuda.sh $JAX_CUDA_VERSION; \
#  fi'

WORKDIR /opt/app

# App non-root user
ENV GROUP=app
ENV USER=flask
ENV UID=12345
ENV GID=23456
RUN useradd -m --no-log-init --system  --uid ${UID} ${USER} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# RUN addgroup --gid "$GID" "$GROUP" \
#   && adduser --uid "$UID" \
#     --disabled-password \
#     --gecos "" \
#     --ingroup "$GROUP" \
#     "$USER"

# Switch to the non-root user
USER "$USER"
ENV PATH="/home/$USER/.local/bin:${PATH}"

# Copy requirements file to our container, install, and remove
# files to we don't need to reduce the container size
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
COPY requirements.txt .
RUN pip install \
    --no-cache-dir \
    --no-warn-script-location \
    --user \
    -r requirements.txt 
  # && find "/home/$USER/.local" \
  #   \( -type d -a -name test -o -name tests \) \
  #   -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
  #   -exec rm -rf '{}' +

# Copy app to container (with privileges to non-root user)
#COPY --chown=$USER:$GROUP . .
COPY . . 
RUN sudo chmod -R a+rwx . 
# COPY model-1reghx5l:v9 model-1reghx5l:v9
# COPY vqgan_weights . 
# COPY clip_weights . 
# COPY clip_processor_weights . 
# Gunicorn is run from the docker-compose file
