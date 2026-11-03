FROM python:3.12-slim

WORKDIR /grasp
ENV PYTHONUNBUFFERED=1 \
  GRASP_INDEX_DIR=/opt/grasp

# Triton compiles kernels at runtime and needs a C compiler
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc libc6-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install GRASP with all dependencies pinned to known-good versions
RUN pip install --no-cache-dir -c constraints.txt ".[evaluation,grisp]"

# Run GRASP by default; override flags via `docker run grasp -- <args>`
ENTRYPOINT ["grasp"]
CMD ["--help"]
