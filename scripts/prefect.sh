#!/usr/bin/env bash

. .env

CMD="docker compose run --rm prefect-cli python -m flows"

${CMD}.generate_data
${CMD}.fit_surrogate
${CMD}.optimise_surrogate
