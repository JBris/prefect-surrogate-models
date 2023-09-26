#!/usr/bin/env bash

. .env

docker compose down
docker compose pull
docker compose build
docker compose up -d 