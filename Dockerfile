# Container image for running Love Predictor on any Docker host
FROM python:3.13-slim

# Run as a normal user instead of root, for safety
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Install packages first so this layer is cached when only the code changes.
# The CPU-only build of PyTorch is much smaller than the default GPU build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Download the language model at build time so the app starts quickly
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
# The model is now inside the image, so never contact the Hugging Face website at runtime
ENV HF_HUB_OFFLINE=1

COPY --chown=user . .

# Train and save the prediction model so the first start doesn't have to
RUN python -m model.love_predictor

# Real answers from the story page are saved here. Mount a volume on it to keep them
# when the container is replaced: docker run -v love-data:/home/user/app/data ...
RUN mkdir -p data

# Listen on every network interface, not just localhost, so other devices on your
# local network (phones, laptops) can open the app through the published port.
# Same port as running locally; hosting services can override it with the PORT variable
ENV HOST=0.0.0.0 \
    PORT=5001
EXPOSE 5001

# One worker keeps a single copy of the language model in memory; threads let predictions
# answer while tips are generating. The long timeout covers slow tip generation on CPU.
# "exec" makes gunicorn the main process, so it shuts down cleanly on "docker stop"
CMD ["sh", "-c", "echo \"Love Predictor: open http://localhost:$PORT on this computer, or http://<this computer's IP>:$PORT on other devices on your network\" && exec gunicorn --bind $HOST:$PORT --workers 1 --threads 4 --timeout 300 main:app"]
