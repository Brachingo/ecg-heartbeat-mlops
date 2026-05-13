#!/bin/bash

# Build the Docker image
docker build -t ecg-heartbeat-mlops .

# Run the Docker container
docker run -it --rm -p 8000:8000 ecg-heartbeat-mlops