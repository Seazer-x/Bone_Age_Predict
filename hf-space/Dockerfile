FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 --branch v1.0.0 https://github.com/Seazer-x/Bone_Age_Predict.git .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "Bone-pre.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
