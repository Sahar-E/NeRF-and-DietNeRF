#!/bin/bash

# This script was used to build and tag

docker build . -t $1
docker tag $1 gcr.io/$2/$1
docker push gcr.io/$2/$1
