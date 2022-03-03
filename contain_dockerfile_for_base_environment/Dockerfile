FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-7:latest
WORKDIR /

RUN pip install tensorflow-graphics==1.0.0
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install cloudpathlib[gs]
RUN pip install numpy-quaternion

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3"]

