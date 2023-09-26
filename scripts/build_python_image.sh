#!/usr/bin/env bash

. .env

cd ./services/python

docker compose pull
docker compose build

echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

DOCKER_IMAGE_HASH=$(docker images --format "{{.ID}} {{.CreatedAt}}" | sort -rk 2 | awk 'NR==1{print $1}')
docker tag "$DOCKER_IMAGE_HASH" "$GITHUB_CONTAINER_REPO"
docker push "$GITHUB_CONTAINER_REPO"