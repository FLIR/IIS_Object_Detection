# This includes the CAFFE SSD-Mobilenet scripts for training customized applications

An example on training a QR-code detector is provided [here](https://confluencecommercial.flir.com/display/IISRT/%5BSimplified%5D+QR+Code+Localization+Development+Process) to illustrate the development process. 

## Training Environment Setup
**[Optional:] If you have Caffe-SSD with GPU (CUDA) installed in your Linux machine you can skip the following steps and go directly to File Structure in section 5.**

Here, we use caffe framework to train the object detection model. You will need to have the Caffe-SSD environment setup on you device. Please make you have the environment setup before proceeding. 

In this tutorial, we use Docker to build the Caffe-ssd environment. However, you will need to install Docker with GPU support (Cuda) before you can build a docker image. You can find a helpful procedure to install Docker/Cuda on Ubuntu-18.04 OS [here](https://confluencecommercial.flir.com/display/IISRT/Installation+guide+for+Docker-ce%2C+Cuda+and+Nvidia-drivers).

### Build Docker Image
Build a docker image with Caffe-SSD, Opencv-3.4, and CUDA-8 GPU drive support.

* Download the docker.zip file from [here](https://confluencecommercial.flir.com/display/IISRT/Installation+guide+for+Docker-ce%2C+Cuda+and+Nvidia-drivers).
This contains the caffe-ssd.dockerfile and Makefile.config files.
* Run the docker build command and note the following:
Run the docker build command inside the same directory as the caffe-ssd.dockerfile and Makefile.config files.
```bash
docker build -f caffe-ssd.dockerfile -t caffe-ssd/opencv3.4:latest-devel-cuda8-cudnn7-py3.5-ubuntu16.04-ch .
```

### Run Docker Environment
Run docker container using the caffe-ssd image as follows

```bash
docker run --gpus all -it --rm --name caffe-ssd-opencv3-latest-ch  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 -v /dev:/dev -v /home/research/YourFolder:/home/docker/YourFolder -e DISPLAY=${DISPLAY} caffe-ssd/opencv3.4:latest-devel-cuda8-cudnn7-py3.5-ubuntu16.04-ch
```

Confirm correct caffe-ssd build by running the following command inside the caffe-ssd docker container enviroment.
```bash
cd /opt/caffe && make runtest
```


## File Structure
This provides an overview of your file structure expectations. Place your files in the following fashion to work with the scripts provided in this tutorial. 

Below is the expected dataset structure.

DATASETS     
└── your_dataset
   ├── original
   │   ├── Images
   │   │  ├── image1.jpg
   │   │  ├── image2.jpg
   │   │  ├── image3.jpg
   │   │   :
   │   │   :
   │   │   :   
   │   │  └── imageN.jpg
   │   └── Annotations
   │      ├── image1.xml
   │      ├── image2.xml
   │      ├── image3.xml
   │       :
   │       :
   │       :   
   │      └── imageN.xml
   └── augmented
       ├── Images
       │  ├── image1.jpg
       │  ├── image2.jpg
       │  ├── image3.jpg
       │   :
       │   :
       │   :   
       │  └── imageN.jpg
       ├── Annotations
       │  ├── image1.xml
       │  ├── image2.xml
       │  ├── image3.xml
       │   :
       │   :
       │   :   
       │  └── imageN.xml
       ├── ImageSets
       │  └── Main
       │     ├── trainval.txt
       │     └── test.txt
       └── lmdb
           ├── trainval_lmdb
           │   ├── data.mdb
           │   └── lock.mdb
           └── test_lmdb 
                ├── data.mdb
                └── lock.mdb


Below is the structure of the data preparation and training scripts. 

caffe_ssd     
├── README.md
├── project.config
├── run.sh
├── template
│  ├── data
│  │  ├── data_partition.sh
│  │  ├── create_list.sh
│  │  └── create_data.sh
│  └── MobileNet-SSD 
│      ├── mobilenet_iter_73000.caffemodel
│      ├── gen.py   
│       :
│       :
│       :
│      ├── train.sh
│      └── example
└── PROJECT_NAME (Your project scripts and logs will be created and saved under this folder)
   ├── MobileNetSSD_train.prototxt
   ├── MobileNetSSD_test.prototxt
   ├── MobileNetSSD_deploy.prototxt
   ├── train.prototxt
   ├── deploy.prototxt
   ├── solver_train.prototxt
   ├── solver_test.prototxt
   ├── train.sh
   └── test.sh 


## Train Your Own Model
### Download Training Scripts
Download MobileNet-SSD training scripts, including the pretrained model weights, from the FLIR repo here.  These scripts are modified based on the original [MobileNet-SSD repo](https://github.com/chuanqi305/MobileNet-SSD). 

```bash
git clone https://bitbucketcommercial.flir.com/scm/rdl/caffe_ssd.git
```
A 'MobileNet-SSD' folder is included under 'caffe_ssd/template/' with the code from the original MobileNet-SSD repo for testing and retraining. 

### Train
Lightning Memory-Mapped Database (LMDB) is a software library that provides a high-performance embedded transactional database in the form of a key-value store.

#### Define configuration file in "caffe_ssd/project.config"
First, `cd caffe_ssd`. Modify project.config file for your project.  

```bash
DATA_DIR="DATASETS/your_dataset/augmentated"  # Your input data folder.
IMAGE_FORMAT='.png'     # Only one image format is allowed for all images. Please convert your images into the same format, i.e., one of 'jpg', 'png', and 'bmp' format.
TEST_SET_PERCENTAGE=15  # Percentage of all images that goes to test set, the rest will go to trainval set.
PROJECT_NAME="qr"       # A new folder with this name will be created under caffe_ssd/.
CLASSNUM=2              # The number of classes in your dataset. It is also reflected in "labelmap.prototxt" # In the case of the QR code example, there are 2 classes, background and QR. So CLASSNUM is 2.
CLASSES="QR"            # Comma separated class names, Do not need to include the background class.
```

#### Generate LMDB files & train your model

```bash
./run.sh
```
The script 'run.sh' generates LMDB files and calls 'train.sh' for training your own model. The training output files (*_iter_*.caffemodel, *_iter_*.solverstate) are saved under "caffe_ssd/${script_folder_name}/snapshot" directory.
The training can be terminated early if the loss is at a satisfactory level and has stopped decreasing. The latest model weight will be saved.

### Test
Test your trained model and evaluate the result. [Optional]
```bash
cd caffe_ssd/${PROJECT_NAME}
./test.sh
```
Test mAP was reported to be 100%.
