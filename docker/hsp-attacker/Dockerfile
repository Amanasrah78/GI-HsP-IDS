FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nmap \
        hydra \
        medusa \
        mosquitto-clients \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /attack

CMD ["sleep", "infinity"]
