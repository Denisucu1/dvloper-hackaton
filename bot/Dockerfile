FROM python:3.13-slim

WORKDIR /bot

RUN pip install --no-cache-dir "openai>=3,<4"

COPY llm_client.py memory_store.py solver_runner.py prep_engine.py chat_engine.py main.py ./

# -u keeps stdout unbuffered so JSON lines reach the arena immediately
CMD ["python", "-u", "main.py"]
