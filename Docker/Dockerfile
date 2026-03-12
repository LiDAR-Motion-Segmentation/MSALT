FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System Dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-venv git wget sudo \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libx11-xcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
    libxcb-cursor0 \
    libxcb-cursor-dev \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb-xfixes0 \
    libxcb-xfixes0-dev \
    libxcb-xkb1 \
    libxcb-xkb-dev \
    libxcb-keysyms1 \
    libxcb-keysyms1-dev \
    libxcb-icccm4 \
    libxcb-icccm4-dev \
    libxcb-image0 \
    libxcb-render-util0 \
    libxcb-ewmh2 \
    libegl1 \
    libopengl0 \
    libdbus-1-3 \
    adwaita-icon-theme-full \
    gnome-themes-extra \
    qt5-style-plugins \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Setup UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create User (Matches Host for file permissions)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} msalt-user && \
    useradd -m -u ${USER_ID} -g msalt-user -s /bin/bash msalt-user && \
    echo "msalt-user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
WORKDIR /app

# Install Deps
# We copy ONLY configuration files first to cache dependencies
COPY pyproject.toml uv.lock ./
RUN chown -R msalt-user:msalt-user /app

# Switch to user for installation
USER msalt-user
# Create venv and install deps
RUN uv sync --frozen --no-install-project

# Download Weights (Cached in image)
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -O sam2_b.pt || true
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt -O yolo26l.pt || true