FROM tensorflow/tensorflow:2.21.0-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1

# Sistemske biblioteke
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# TensorFlow + kompatibilne CUDA/cuDNN biblioteke
RUN python -m pip install --no-cache-dir \
    "tensorflow[and-cuda]==2.21.0"

# Biblioteke za projekat
RUN python -m pip install --no-cache-dir \
    "numpy<2" \
    "opencv-python-headless==4.11.0.86" \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    imageio \
    scikit-image \
    h5py \
    kagglehub

WORKDIR /workspace

CMD ["/bin/bash"]