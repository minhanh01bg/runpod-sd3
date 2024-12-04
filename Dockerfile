# Base image
FROM python:3.10-slim

ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir diffusers["torch"] transformers
RUN pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -i https://test.pypi.org/simple/ bitsandbytes

RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# RUN pip install bitsandbytes accelerate
# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python /cache_models.py && \
    rm /cache_models.py

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /rp.handler.py
