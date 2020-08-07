# This includes the CAFFE SSD-Mobilenet scripts for training customized applications

## Training Environment Setup
[Optional:] If you have Caffe-SSD with GPU (CUDA) installed in your Linux machine you can skip the following steps and go directly to File Structure in section 5.

Here, we use caffe framework to train the object detection model. You will need to have the Caffe-SSD environment setup on you device. Please make you have the environment setup before proceeding. 

In this tutorial, we use Docker to build the Caffe-ssd environment. However, you will need to install Docker with GPU support (Cuda) before you can build a docker image. You can find a helpful procedure to install Docker/Cuda on Ubuntu-18.04 OS here.

### Build Docker Image
Build a docker image with Caffe-SSD, Opencv-3.4, and CUDA-8 GPU drive support.

Download the docker.zip file from here.
This contains the caffe-ssd.dockerfile and Makefile.config files.
Run the docker build command and note the following:
Run the docker build command inside the same directory as the caffe-ssd.dockerfile and Makefile.config files.
docker build -f caffe-ssd.dockerfile -t caffe-ssd/opencv3.4:latest-devel-cuda8-cudnn7-py3.5-ubuntu16.04-ch .
Run Docker Environment
Run docker container using the caffe-ssd image as follows

docker run --gpus all -it --rm --name caffe-ssd-opencv3-latest-ch  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 -v /dev:/dev -v /home/research/chase:/home/docker/chase -e DISPLAY=${DISPLAY} caffe-ssd/opencv3.4:latest-devel-cuda8-cudnn7-py3.5-ubuntu16.04-ch

Confirm correct caffe-ssd build by running the following command inside the caffe-ssd docker container enviroment.
cd /opt/caffe && make runtest

