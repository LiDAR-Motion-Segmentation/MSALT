FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Dependencies (Python + GUI libs + OpenCV)
# We need libgl1 and libx11 for PyQt and Open3D to render
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies
# --system installs into the system python, avoiding venv complexity inside Docker
RUN uv sync --system --frozen
COPY . .

CMD ["uv", "run", "main.py"]