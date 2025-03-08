FROM tensorflow/tensorflow:2.13.0-gpu

# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    libosmesa6-dev \
    git \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip 
    # && \ python3 -m pip install --no-cache-dir --upgrade open3d==0.17.0


##############################################################
####                     Source Code                      ####
##############################################################
COPY requirements.txt /tmp
# RUN pip install -r /tmp/requirements.txt

RUN pip install grpcio protobuf --upgrade

RUN pip install opencv-python-headless

RUN mkdir /workspace
COPY insertion /workspace/insertion
COPY qbit/interfaces /workspace/qbit/interfaces
COPY setup.py /workspace
COPY pyproject.toml /workspace

WORKDIR /workspace


# Configure Mesa EGL for headless rendering
ENV EGL_PLATFORM=surfaceless
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

EXPOSE 50052
CMD ["python", "/workspace/qacbi/grpc/inference_server.py"]
