ARG CUDA_IMAGE="12.5.0-devel-ubuntu22.04"
FROM nvcr.io/nvidia/cuda:${CUDA_IMAGE}

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.6 8.9"

RUN apt-get update && apt-get upgrade -y && apt-get install git cmake curl -y
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -L -o /tmp/Miniconda3-latest-Linux-x86_64.sh
RUN bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init
RUN . ~/.bashrc && conda install -c conda-forge libstdcxx-ng
RUN . ~/.bashrc && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN . ~/.bashrc && pip install packaging ninja cpufeature numpy
RUN . ~/.bashrc && pip install flash-attn==2.7.4.post1 --no-build-isolation
RUN . ~/.bashrc && pip install flashinfer-python==0.2.3
RUN mkdir -p /opt/ml/code
RUN git clone https://github.com/kvcache-ai/ktransformers.git /opt/ml/code/ktransformers
RUN . ~/.bashrc && cd /opt/ml/code/ktransformers && git submodule init && git submodule update && bash install.sh

RUN curl https://github.com/peak/s5cmd/releases/download/v2.0.0/s5cmd_2.0.0_Linux-64bit.tar.gz -L -o /tmp/s5cmd.tar.gz
RUN mkdir -p /tmp/s5cmd && tar -xvf /tmp/s5cmd.tar.gz -C /tmp/s5cmd
RUN cp /tmp/s5cmd/s5cmd /opt/ml/code

COPY backend/ktransformers/DeepSeek-R1/ /opt/ml/model/DeepSeek-R1c

# Ensure the serve script has executable permissions
# RUN chmod +x /usr/bin/serve
RUN . ~/.bashrc && pip install boto3 fastapi uvicorn requests openai hf_transfer modelscope

# Expose port 8080
EXPOSE 8080
WORKDIR /opt/ml/code

# Set the serve script as the entrypoint
ENTRYPOINT ["/usr/bin/serve"]