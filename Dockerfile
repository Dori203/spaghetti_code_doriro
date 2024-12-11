#FROM pytorch/pytorch:latest
#FROM nvcr.io/nvidia/pytorch:23.12-py3  #this is what eyal set and worked.
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG CACHEBUST=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
WORKDIR /workspace

# System dependencies
RUN apt-get update && \
    apt-get purge -y hwloc-nox libhwloc-dev libhwloc-plugins && \
    apt-get autoremove -y && \
    apt-get install -y \
    git \
    libvtk7.1 \
    libgl1-mesa-glx \
    tzdata \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and create environment
#COPY environment-docker.yml .
#RUN conda env create -f environment-docker.yml && \
#    conda clean -afy

RUN pip install flask flask-cors trimesh numpy open3d Pillow pyglet

# Activate environment
#SHELL ["conda", "run", "-n", "new_env", "/bin/bash", "-c"]
RUN pip install runpod libigl
RUN pip install pickle5
RUN pip install h5py scikit-image

# Clone repository
#RUN git clone https://github.com/dori203/spaghetti_code_doriro.git spaghetti_code_doriro
#RUN --mount=type=cache,target=/root/.cache/pip \
#    echo "Updating at $(date)" && \
#    git clone https://github.com/dori203/spaghetti_code_doriro.git spaghetti_code_doriro && \
#    cd spaghetti_code_doriro && \
#    git pull
RUN echo "Fetching latest at ellie $(date)" && \
    git clone https://github.com/dori203/spaghetti_code_doriro.git spaghetti_code_doriro && \
    cd spaghetti_code_doriro && \
    git pull

WORKDIR /workspace/spaghetti_code_doriro

RUN echo "Copying handler at $(date)"
COPY ./rp_handler.py ./rp_handler.py
COPY ./test_input.json ./test_input.json


CMD ["python", "rp_handler.py"]