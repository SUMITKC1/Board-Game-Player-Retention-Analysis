# Lightweight Python image
FROM python:3.11-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set workdir
WORKDIR /app

# System deps
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file and install first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Default command opens Jupyter Lab for notebooks; can be overridden
EXPOSE 8888
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --no-browser --allow-root"]
