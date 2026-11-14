# The whole bot is stdlib-only: no pip install, nothing to drift, a small and fast image.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/bot

WORKDIR /bot

COPY main.py ./
COPY harness ./harness
COPY logo.jpg /logo.jpg

# No arguments: the arena runs `docker run -i --rm <image>` (rule 5.2).
CMD ["python", "-u", "main.py"]
