FROM python:3.12-slim

WORKDIR /bot

COPY ./requirements.txt /bot
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

# Copy the source directory
COPY ./src /bot/src

CMD ["python", "src/bot.py"]