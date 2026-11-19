# EyeForge: headless Blender 4.2 LTS + Python tooling. Build and run the whole pipeline:
#   docker build -t eyeforge .
#   docker run --rm -v "$PWD:/work" eyeforge            # = ./run_all.sh inside the container
FROM python:3.11-slim-bookworm
ARG BLENDER_VERSION=4.2.23
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl xz-utils libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 \
        libgl1 libegl1 libglib2.0-0 ffmpeg make \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
        | tar -xJ -C /opt && ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV BLENDER_PATH=/usr/local/bin/blender PYTHON=python3
WORKDIR /work
CMD ["./run_all.sh"]
