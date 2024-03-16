FROM nvidia/cuda:12.3.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.11

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

RUN pip install -U pip setuptools


COPY . .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt