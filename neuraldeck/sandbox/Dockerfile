# The image the chat's run_python tool executes code in. Built once by the
# deck (docker build -t neuraldeck-sandbox) and run with no network, a CPU
# and memory cap, a read-only root and a short timeout; nothing in it can
# reach the deck, the proxy or the LAN.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy pandas sympy matplotlib requests && \
    useradd -m -u 65534 -o sandbox 2>/dev/null || true
ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /tmp
USER 65534
