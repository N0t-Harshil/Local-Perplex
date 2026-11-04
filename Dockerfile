FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 PORT=8000 LOCAL_PERPLEX_NO_AUTOSTART=1 \
    LOCAL_PERPLEX_CONFIG=/data/config.json \
    LOCAL_PERPLEX_HISTORY_DIR=/data/history LOCAL_PERPLEX_DOCUMENT_DIR=/data/documents \
    LOCAL_PERPLEX_GRAPH_DIR=/data/graph LOCAL_PERPLEX_SESSIONS_DIR=/data/sessions

WORKDIR /app
COPY pyproject.toml README.md ./
COPY local_perplex ./local_perplex
RUN pip install . && useradd --create-home app && mkdir -p /data && chown app /data
USER app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["perplex-gui"]
