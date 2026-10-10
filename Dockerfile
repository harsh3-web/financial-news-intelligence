FROM python:3.13-slim
WORKDIR /app

COPY requirements-api.txt .
# CPU-only torch (the normal Linux torch adds ~2 GB of GPU code we don't need)
RUN pip install --no-cache-dir $(grep torch requirements-api.txt) --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/finbert/ models/finbert/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]