FROM apache/airflow:2.9.0-python3.10

USER root
COPY requirements.txt /tmp/requirements.txt

# Optional: Install OS-level dependencies if needed
# RUN apt-get update && apt-get install -y build-essential

USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt