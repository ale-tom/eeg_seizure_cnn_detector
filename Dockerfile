# Dockerfile
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies and apply security updates
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install jupyterlab

# Copy project
COPY . .

# Expose Jupyter 
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
CMD ["--NotebookApp.token=''"]
