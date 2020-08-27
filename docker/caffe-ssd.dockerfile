#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

# Add user
ARG USER=docker
ARG UID=1000
ARG GID=1000

# Sudo user password
ARG PW=docker


# Temporary assign user as root to perform apt-get and sudo functions
USER root

RUN useradd -m ${USER} --uid=${UID} &&  echo "${USER}:${PW}" | chpasswd
# This line is optional
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# Install sudo and add user to sudo group
RUN apt-get update
RUN apt-get --allow-insecure-repositories --fix-missing update 


RUN apt-get install -y -q \
	software-properties-common \
    build-essential cmake checkinstall \
    pkg-config \
    wget git curl \
    unzip yasm \
    pkg-config \
    nano vim\
    sudo \
    python-scipy 

# add sudo user 
RUN  adduser ${USER} sudo

## Python libraries 
# TODO: upgrade from python 3.5 to python 3.7.

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y -q python-dev python3-dev

## Intsall pip and pip3 and some python libraries

# Install the latest version of pip (https://pip.pypa.io/en/stable/installing/#using-linux-package-managers)
RUN wget --no-check-certificate  https://bootstrap.pypa.io/get-pip.py

RUN python get-pip.py && pip install numpy scipy
RUN python3 get-pip.py && pip3 install numpy scipy
RUN pip install --upgrade google-api-python-client
RUN pip install scikit-image 


# gflags, glog, protobuf, hdf5, protobuf
RUN apt-get install -y -q libgflags-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev libhdf5-dev


#optional dependencies Image I/O libs
RUN apt-get install -y -q  libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev 


# Parallelism library C++ for CPU
RUN apt-get install -y -q libtbb2 libtbb-dev

#Optimization libraries for OpenCV
RUN apt-get install -y -q libatlas-base-dev gfortran

#Optional libraries: 
RUN apt-get install -y -q libgphoto2-dev libeigen3-dev doxygen

## Required libraries for Caffe

# GTK (gtk3) support for GUI features for the graphical user functionalites
RUN apt-get install -y -q libgtk-3-dev


RUN apt-get update

# lmdb, opencv, and hdf5 
RUN apt-get install -y -q liblmdb-dev libopencv-dev libhdf5-serial-dev libhdf5-dev

# boost, blas, leveldb, snappy
RUN apt-get install -y -q libboost-all-dev libblas-dev libatlas-base-dev libopenblas-dev  libleveldb-dev libsnappy-dev


#######OPENCV 3.4 INSTALLATION##############

# set working directory to /opt to download opencv repo in.
WORKDIR /opt

# set build arguments
ARG CLONE_TAG=3.4
ARG OPENCV_TEST_DATA_PATH=/opt/opencv_extra/testdata

# opencv extra test dataset
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/opencv/opencv_extra.git
# contrib repo
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/opencv/opencv_contrib.git
# opencv repo
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/opencv/opencv.git 

# set env and working directory to opencv root
ENV OPENCV_TEST_DATA_PATH=/opt/opencv_extra/testdata/
ENV OPENCV_ROOT=/opt/opencv
WORKDIR $OPENCV_ROOT


RUN mkdir build && cd build && \
    cmake   -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
            -D PYTHON_EXECUTABLE=$(which python) \
            -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -D PYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
            -D INSTALL_C_EXAMPLES=ON     \
            -D INSTALL_PYTHON_EXAMPLES=ON     \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules     \
            -D BUILD_EXAMPLES=ON  \
            -D BUILD_NEW_PYTHON_SUPPORT=ON  \
            -D BUILD_opencv_python3=ON  \
            -D HAVE_opencv_python3=ON \
            -D BUILD_TIFF=ON \
            -D BUILD_opencv_java=OFF \
            -D WITH_CUDA=OFF \
            -D WITH_OPENGL=ON \
            -D WITH_OPENCL=ON \
            -D WITH_IPP=ON \
            -D WITH_TBB=ON \
            -D WITH_EIGEN=ON \
            -D WITH_V4L=ON \
            -D WITH_QT=OFF \
            -D WITH_GTK=OFF \
            -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 ..


RUN echo "$(nproc)"
RUN cd build && make -j"$(nproc)" && make install
RUN build/bin/opencv_test_core

# Set enviroment variabiles
ENV QT_X11_NO_MITSHM=1

######## CAFFE-SSD INSTALLATION######

RUN apt-get update

# set working directory to /opt
WORKDIR /opt

ARG CLONE_TAG=ssd

RUN git clone -b ${CLONE_TAG} --depth 1  https://github.com/weiliu89/caffe.git && for req in $(cat python/requirements.txt) pydot; do pip install $req; done



ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# copy caffe build config file to /opt/caffe (This is provided sepritly and must be placed in the same directory as this docker file. In addition, you must run the docker build command from the same directory)
COPY Makefile.config $CAFFE_ROOT

# Build and test caffe
RUN make all && make test
# RUN make runtest # Recommend run inside enviroment. This failes during build but passes inside env in current version of dockerfile. 

# Build python-caffe api 
RUN cp build/lib/libcaffe.so* /usr/lib
RUN make pycaffe
RUN cp -r python/caffe/ /usr/local/lib/python3.5/dist-packages/

# Set enviroment variables
ENV PATH=$PATH:/home/docker/.local/bin
ENV PYCAFFE_ROOT=$CAFFE_ROOT/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
ENV PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Setup default user, when enter docker container

USER ${UID}:${GID}
WORKDIR /home/${USER}
