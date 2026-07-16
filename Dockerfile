# ragproof: reproducible offline demo + test run in a container.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ragproof ./ragproof
COPY tests ./tests

RUN pip install --no-cache-dir -e .

# Default: run the offline retrieval-metrics demo. Override to run tests:
#   docker run --rm ragproof python -m pytest -q
CMD ["python", "-m", "ragproof.demo"]
