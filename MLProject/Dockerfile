FROM python:3.12.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY conda.yaml .
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    mlflow==2.19.0 \
    matplotlib==3.9.2 \
    seaborn==0.13.2

COPY . .

EXPOSE 5000

CMD ["python", "modelling.py"]
