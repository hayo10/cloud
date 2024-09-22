FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0
FROM nvcr.io/nvidia/pytorch:23.09-py3
FROM python:3.10-slim



WORKDIR /home/hayoung/cloud/workspace

COPY requirements.txt ./

RUN echo "Acquire::Check-Valid-Until \"false\";\nAcquire::Check-Date \"false\";" | cat > /etc/apt/apt.conf.d/10no--check-valid-until
RUN apt-get update &&  apt-get install -y git

ENV TORCH="http://nvidia.box.com/shared/static/1v2cc4ro6zvsbu0p8h6qcuaqco1qcsif.whl -O torch-1.4.0-cp27-cp27mu-linux_aarch64.whl"
RUN pip3 install --no-cache $TORCH
RUN sudo apt-get install libopenblas-base libopenmpi-dev  

ENV TORCH_INSTALL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+40ec155e58.nv24.03.13384722-cp310-cp310-linux_aarch64.whl"
RUN pip3 install --no-cache $TORCH_INSTALL
ENV FLASH="https://static.abacus.ai/pypi/abacusai/gh200-llm/flash_attn-2.3.6-cp310-cp310-linux_aarch64.whl"
RUN pip3 install --no-cache $FLASH
RUN pip install --no-cache -r requirements.txt

UN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn

ENV CUDA_HOME /usr/local/cuda-12.2
ENV PATH /usr/local/cuda-12.2/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

COPY . .

CMD  ["python", "script.py"]