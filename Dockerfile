FROM python:3.12-slim

WORKDIR /grasp
ENV PYTHONUNBUFFERED=1 \
  GRASP_INDEX_DIR=/opt/grasp

# Install C build toolchain (required for search-rdf Rust/maturin compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install GRASP
RUN pip install --no-cache-dir .

# Run GRASP by default; override flags via `docker run grasp -- <args>`
ENTRYPOINT ["grasp"]
CMD ["--help"]
