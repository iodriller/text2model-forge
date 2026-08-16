# The control plane intentionally stays separate from GPU-heavy ComfyUI/PyTorch.
# Compose connects it to Ollama and to an optional host ComfyUI service.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Asset Forge resolves adapters and other resources relative to the checkout,
# so retain the full source tree and use an editable install inside the image.
COPY . .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 assetforge \
    && useradd --uid 10001 --gid assetforge --no-create-home assetforge \
    && mkdir -p /workspace \
    && chmod 0777 /workspace

USER assetforge
EXPOSE 8766
VOLUME ["/workspace"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/doctor', timeout=2).read(1)"]

CMD ["python", "-m", "darkness", "studio", "--workspace", "/workspace", "--host", "0.0.0.0", "--port", "8766", "--allow-non-loopback"]
