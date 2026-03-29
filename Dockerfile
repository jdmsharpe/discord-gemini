ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

WORKDIR /bot

COPY ./requirements.txt /bot
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

# Copy the source directory
COPY ./src /bot/src

CMD ["python", "src/bot.py"]
