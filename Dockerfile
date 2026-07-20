FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY Agents/ Agents/
COPY engine/ engine/
COPY backend/ backend/
COPY frontend/ frontend/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .
ENV HOST=0.0.0.0
EXPOSE 5000
CMD ["python", "backend/ui/server.py"]
