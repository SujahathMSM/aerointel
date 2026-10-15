FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

# Optional OCR for scanned PDFs: `docker compose build --build-arg INSTALL_OCR=true`
# (or INSTALL_OCR=true in .env), then run with OCR_ENABLED=true. Off by default
# because it adds the tesseract program and PyMuPDF to the image.
ARG INSTALL_OCR=false
COPY requirements-ocr.txt ./
RUN if [ "$INSTALL_OCR" = "true" ]; then apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir -r requirements-ocr.txt; fi

USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]