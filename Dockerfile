# This is the Dockerfile that wrapes the project to run in GCP.
# It is intended to run in a GCP environment workbench.
# Please remeber to connect the correct bucket to save your results in the Yaml config file.

# TODO: Please insert your bucket name:
FROM gcr.io/<YOUR_BUCKET_NAME>/base_env_gpu_tf_2.7:latest
WORKDIR /

# Copies the trainer code to the docker image.
COPY Assets /Assets
COPY config_files /config_files
COPY src /src
COPY main.py /main.py
#COPY Results /Results


# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "-m", "main"]

