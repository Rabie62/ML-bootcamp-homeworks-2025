FROM agrigorev/zoomcamp-model:2025
WORKDIR /code
COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv && uv sync
COPY pipeline_v1.bin .
COPY main.py .
CMD ["/code/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]