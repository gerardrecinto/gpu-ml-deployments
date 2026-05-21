FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
        "torch==1.12.1+cu116" \
        "torchvision==0.13.1+cu116" \
        --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /workspace
COPY train.py .

CMD ["python3", "train.py"]
