FROM python:3.10-slim-bullseye
# FROM continuumio/miniconda3

WORKDIR /

COPY Real-ESRGAN Real-ESRGAN
COPY requirements.txt requirements.txt
# COPY conda_requirements.txt conda_requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN python3 -m pip install basicsr facexlib gfpgan

COPY enhance_images.py enhance_images.py
# RUN conda create -n image-enhance python=3.0 --file=conda_requirements.txt -c conda-forge

WORKDIR /Real-ESRGAN/
RUN python3 -m pip install -r requirements.txt
RUN python3 setup.py develop

WORKDIR /

CMD "bin/bash"
