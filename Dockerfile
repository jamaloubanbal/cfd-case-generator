# CFD Case Generator: the web app with OpenFOAM, ready to run cases from the browser.
#
#   docker build -t cfd-case-generator .
#   docker run --rm -p 8501:8501 -e GEMINI_API_KEY=... cfd-case-generator   # → http://localhost:8501
#
# OpenFOAM v1912 from the Ubuntu 24.04 package: the version the CI tests and validation use.
# code_saturne is not included (no Ubuntu package); the app then offers OpenFOAM only.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Open MPI in containers: no cross-memory attach, which Docker forbids by default
    OMPI_MCA_btl_vader_single_copy_mechanism=none

RUN apt-get update \
 && apt-get install -y --no-install-recommends openfoam openmpi-bin python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app app
COPY templates templates
COPY examples examples
COPY docker/entrypoint.sh /usr/local/bin/cfd-entrypoint
RUN chmod +x /usr/local/bin/cfd-entrypoint

# Run as the image's unprivileged user (uid 1000, as Hugging Face Spaces expects):
# mpirun also refuses to run as root.
USER ubuntu
ENV HOME=/home/ubuntu \
    PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

ENTRYPOINT ["cfd-entrypoint"]
CMD ["web"]
