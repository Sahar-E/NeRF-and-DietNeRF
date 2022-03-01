#!/bin/bash

# This script can launch a GCP (google cloud platform) Vertex AI instance by simply activating it with a name of your
# choice.

# Example:
# sh custom_job_launcher.sh 256px_alexander_71pics_sphere_dietnerf_use10pics:4.0
#
# Where 256px_alexander_71pics_sphere_dietnerf_use10pics:4.0 will be the name of the docker, and the job name.
#
# The script should be placed in the root of the project directory, with the script "docker_maker.sh".

#TODO: Insert your bucket name here
BUCKET_NAME=INSERT_BUCKET_NAME

printf "Bulding the docker $1\n"
sh docker_maker.sh $1 $BUCKET_NAME
printf "Done bulding the docker $1\n"

printf "Creating new gcloud_config.yaml\n"
printf "\
workerPoolSpecs:   \n\
- containerSpec:   \n\
    imageUri: gcr.io/$BUCKET_NAME/$1   \n\
  diskSpec:   \n\
    bootDiskSizeGb: 100   \n\
    bootDiskType: pd-standard   \n\
  machineSpec:   \n\
    acceleratorCount: 1   \n\
    acceleratorType: NVIDIA_TESLA_A100   \n\
    machineType: a2-highgpu-1g   \n\
  replicaCount: '1' \n\
" > gcloud_config.yaml

printf "Deploying gcloud ai custom-job...\n"
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=$1 \
  --config=gcloud_config.yaml

printf "Deployed gcloud ai custom-job\n"