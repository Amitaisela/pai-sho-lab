FROM python:3.11-slim
WORKDIR /app

# Rust toolchain for building engine/RustEngine/crates/pybind (the PyO3 bridge) via maturin.
# curl + a C linker (gcc, from build-essential) are rustup/cargo's own prerequisites on Debian.
RUN apt-get update \
 && apt-get install --no-install-recommends -y curl build-essential \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

COPY pyproject.toml requirements.txt ./
COPY Agents/ Agents/
COPY engine/ engine/
COPY backend/ backend/
COPY frontend/ frontend/
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e . \
 && pip install --no-cache-dir maturin \
 && (cd engine/RustEngine/crates/pybind && maturin build --release) \
 && pip install --no-cache-dir engine/RustEngine/target/wheels/*.whl
ENV HOST=0.0.0.0
EXPOSE 5000
CMD ["python", "backend/ui/server.py"]
