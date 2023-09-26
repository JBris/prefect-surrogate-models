ARG PYTHON_TAG

FROM python:${PYTHON_TAG}

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev graphviz  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/downloaded_packages

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
