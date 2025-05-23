FROM mcr.microsoft.com/devcontainers/base:focal

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################
# Install libraries for rendering
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update \
    && apt install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    mesa-utils \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev &&\
    rm -rf /var/lib/apt/lists/*

# Install pip and config python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update \
    && apt install -y -qq --no-install-recommends \
    python3-pip python3-dev


##############################################################################
##                               Libraries                                  ##
##############################################################################
WORKDIR /opt/devcon_lib
# ToTg
RUN git clone https://github.com/djumpstre/py-totg.git \
    && pip3 install pybind11[global] \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt install -y -qq --no-install-recommends \
    libeigen3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/devcon_lib/py-totg \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && pip3 install ..

# Track IK
RUN git clone https://github.com/mjd3/tracikpy.git \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt install -y -qq --no-install-recommends \
    libeigen3-dev \
    liborocos-kdl-dev \
    libkdl-parser-dev \
    liburdfdom-dev \
    libnlopt-dev \
    libnlopt-cxx-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*
RUN cd /opt/devcon_lib/tracikpy \
    && pip install .


##############################################################
####                     Source Code                      ####
##############################################################
# # Python pip
# RUN apt-get update && \
#     apt -y install python3-pip && \
#     rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir grpcio protobuf --upgrade
RUN pip install --no-cache-dir opencv-python-headless
# RUN pip install --no-cache-dir loguru open3d imageio

WORKDIR /workspace

# source
COPY qacbi/ /workspace/qacbi
COPY examples/ /workspace/examples
COPY setup.py /workspace
COPY pyproject.toml /workspace

RUN cd /workspace && pip install -e .

# ENV EGL_PLATFORM=surfaceless

# executable
# ENTRYPOINT [ "/entrypoint.sh" ]
CMD python3 /workspace/examples/cloud_perf_sim_grpc.py
