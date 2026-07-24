FROM python:3.12-slim

WORKDIR /grasp
ENV PYTHONUNBUFFERED=1 \
  GRASP_INDEX_DIR=/opt/grasp

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg &&\
    rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install GRASP
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Run GRASP by default; override flags via `docker run grasp -- <args>`
ENTRYPOINT ["grasp"]
CMD ["--help"]
