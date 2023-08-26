FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY .. /app

RUN apk update && apk add build-base libffi-dev python3-dev
RUN apk add py3-cffi

RUN pip install --no-cache-dir -r requirements.txt
