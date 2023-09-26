#!/usr/bin/env bash

. .env

docker compose run --rm prefect-cli "$@"
