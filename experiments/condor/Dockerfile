FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 libgomp1 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.3.3 \
    scikit-learn==1.7.2 \
    rdkit==2025.9.3 \
    datasail==1.3.0

ENV MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace
