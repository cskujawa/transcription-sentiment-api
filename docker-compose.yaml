version: '3.8'
services:
  transcription-sentiment-fastapi:
    hostname: transcription-sentiment-fastapi
    container_name: transcription-sentiment-fastapi
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend:/usr/src/app
    networks:
      - proxy-smart-tilde
    ports:
      - "8000:8000"
