# The control plane intentionally stays separate from GPU-heavy ComfyUI/PyTorch.
# Compose connects it to Ollama and to an optional host ComfyUI service.
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

ARG VETTEDMESH_SOURCE_REVISION=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VETTEDMESH_SOURCE_REVISION=${VETTEDMESH_SOURCE_REVISION}

LABEL org.opencontainers.image.source="https://github.com/iodriller/vettedmesh" \
    org.opencontainers.image.revision=${VETTEDMESH_SOURCE_REVISION}

WORKDIR /app

# VettedMesh resolves adapters and other resources relative to the checkout,
# so retain the full source tree and use an editable install inside the image.
COPY . .
RUN python -m pip install --no-cache-dir --upgrade "pip==26.2.1" "setuptools==84.0.0" \
    && python -m pip install --no-cache-dir --require-hashes -r requirements.lock \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation -e . \
    && groupadd --gid 10001 vettedmesh \
    && useradd --uid 10001 --gid vettedmesh --no-create-home vettedmesh \
    && mkdir -p /workspace \
    && chmod 0777 /workspace

USER vettedmesh
EXPOSE 8766
VOLUME ["/workspace"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/doctor', timeout=2).read(1)"]

CMD ["python", "-m", "darkness", "studio", "--workspace", "/workspace", "--host", "0.0.0.0", "--port", "8766", "--allow-non-loopback"]
