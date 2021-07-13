# docker run --gpus all --rm -it --name caffe-env-1  -e DISPLAY=${DISPLAY}  --net=host  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  -v /dev:/dev -v path/to/host_target_directory:/home/docker asigiuk/caffe-ssd_devel:latest
docker run --gpus all --rm -it --name caffe-env-1  -e DISPLAY=${DISPLAY}  --net=host  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  -v /dev:/dev -v path/to/host_target_directory:/home/docker workingtaechqie/caffe-ssd-bionic-devel:20210713

