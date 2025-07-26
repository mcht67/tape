# Copyright 2024 tu-studio
# This file is licensed under the Apache License, Version 2.0.
# See the LICENSE file in the root of this project for details.

# Use an official Debian runtime with fixed version as a parent image
FROM debian:12-slim

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    curl \
    libbz2-dev \
    git \
    python3-pip \
    openssh-client \
    rsync \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    # Remove apt cache
    && rm -rf /var/lib/apt/lists/*


# Copy the global.env file
COPY global.env /tmp/global.env

# Install Python Version
RUN . /tmp/global.env \
    && echo "Using Python version: ${TUSTU_PYTHON_VERSION}" \
    && echo "Downloading Python version: ${TUSTU_PYTHON_VERSION}" \
    && wget --no-check-certificate https://www.python.org/ftp/python/${TUSTU_PYTHON_VERSION}/Python-${TUSTU_PYTHON_VERSION}.tgz \
    && tar -xf Python-${TUSTU_PYTHON_VERSION}.tgz \
    && cd Python-${TUSTU_PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    # Delete the unzipped directory and downloaded archive to save space
    && rm -rf Python-${TUSTU_PYTHON_VERSION} Python-${TUSTU_PYTHON_VERSION}.tgz \
    # Create symlink for python3
    && ln -s /usr/local/bin/python${TUSTU_PYTHON_VERSION%.*} /usr/local/bin/python3 \
    && ln -s /usr/local/bin/python${TUSTU_PYTHON_VERSION%.*} /usr/local/bin/python

# Set the working directory
WORKDIR /home/app

# Create virtual environment
RUN python3 -m venv /opt/birdset-env
ENV PATH="/opt/birdset-env/bin:$PATH"

# Install your requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Verify ruamel.yaml is installed
RUN python -c "import ruamel.yaml; print('ruamel.yaml installed successfully')"

# Then install BirdSet
RUN git clone --no-recurse-submodules https://github.com/DBD-research-group/BirdSet.git /tmp/birdset && \
    pip install -e /tmp/birdset && \
    rm -rf /tmp/birdset

# Final verification
RUN python -c "import ruamel.yaml; import datasets; print('All packages working')"

RUN rm requirements.txt

ENV PYTHONPATH="/home/app/source"



