FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Some RUN statements are combined together to make Docker build run faster.
# Get latest package listing, install software-properties-common, git, wget,
# compilers and libraries.
# git is required for pyproject.toml toolchain's use of CMakeLists.txt.
# gcc, g++, make are required for compiling and AlphaFold 3 libraries.
# zlib is a required dependency of AlphaFold 3.
RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet wget gcc g++ cmake \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Get apt repository of specific Python versions. Then install Python. Tell APT
# this isn't an interactive TTY to avoid timezone prompt when installing.
RUN add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet python3.9 python3.9-venv

# Create virtual environment
RUN python3.9 -m venv /swbind_venv

# Set environment paths
ENV PATH="/hmmer/bin:/mmseqs/bin:/swbind_venv/bin:$PATH"

# Copy and extract HMMER
COPY hmmer.tar.gz /tmp/hmmer.tar.gz
RUN mkdir -p /hmmer && \
    tar -xzf /tmp/hmmer.tar.gz -C /hmmer --strip-components=1 && \
    rm /tmp/hmmer.tar.gz

# Copy and extract MMseqs2
COPY mmseqs.tar.gz /tmp/mmseqs.tar.gz
RUN mkdir -p /mmseqs && \
    tar -xzf /tmp/mmseqs.tar.gz -C /mmseqs --strip-components=1 && \
    rm /tmp/mmseqs.tar.gz

# Copy the source code from the local machine to the container and
# set the working directory to there.
COPY . /app/swbind
WORKDIR /app/swbind

# Update pip to the latest version and install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r dev-requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-deps .
RUN rm -rf ./src

# Create symbolic links for CUDA libraries
RUN (cd /swbind_venv/lib/python3.9/site-packages/nvidia/cudnn/lib && ln -s ./libcudnn.so.* libcudnn.so); \
    (cd /swbind_venv/lib/python3.9/site-packages/nvidia/cublas/lib && ln -s ./libcublas.so.* libcublas.so)

# Set library path for PaddlePaddle
ENV LD_LIBRARY_PATH="/swbind_venv/lib/python3.9/site-packages/nvidia/cudnn/lib/:/swbind_venv/lib/python3.9/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

# Set work directory
WORKDIR /app/swbind

# Default command
CMD ["/bin/bash"]
