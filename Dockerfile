FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python - <<'PY'
import os
import sys
from urllib.request import urlopen

port = os.environ.get("PORT", "8000")
url = f"http://localhost:{port}/health"

try:
    with urlopen(url, timeout=2) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
PY

CMD ["python", "-m", "server.main"]
