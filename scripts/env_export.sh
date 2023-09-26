#!/usr/bin/env bash

pip list --format=freeze > requirements.txt && \
    mv requirements.txt ./services/python  