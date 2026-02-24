FROM tensorflow/tensorflow:2.15.0-gpu
RUN pip install pandas scikit-learn matplotlib seaborn imageio scikit-image "numpy<2" opencv-python


RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace