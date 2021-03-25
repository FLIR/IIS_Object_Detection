# This includes the CAFFE SSD-Mobilenet scripts for training customized applications

An example on training a QR-code detector is provided [here](https://confluencecommercial.flir.com/display/IISRT/%5BSimplified%5D+QR+Code+Localization+Development+Process) to illustrate the development process.

## Environment Setup

We use caffe framework to train the object detection model. You will need to have the Caffe-SSD environment setup on you device. Please make you have the environment setup before proceeding. 

### Windows Setup

[eric612](https://github.com/eric612/MobileNet-SSD-windows) has an extensive guide to install caffe for Windows, we will need the executables built there for preparing for Training and Training

For simplicity I've created a fork and modified it to utilize CUDA

1. Install CUDA 10 from the Nvidia website and CUDNN v7.6.5.32 (for compatibility with Tensorflow-GPU 1.15 training please use cuda 10.0 and cudnn 7.4)
2. Install Anaconda and create a python3.5 environment or virtual environment
3. Download/Clone [eric612](https://github.com/eric612/MobileNet-SSD-windows) repository or the [forked](https://github.com/dmchong/MobileNet-SSD-windows) repository;
4. Open either a powershell terminal or a command prompt

```bash
cmd # if youre on powershell
git clone https://github.com/dmchong/MobileNet-SSD-windows # if this doesn't work you will need to install git.
cd MobileNet-SSD-windows\scripts
activate python3.5_environment
build_win.cmd # verify that -DCUDNN_ROOT pointing to the correct place
# make sure to remove the build folder before trying again in event of a hiccups
```

5. Get a coffee and wait till the build is complete
6. After its done clone this repository

<!--
### Run Docker Environment
**This section is optional:**
**If you have Caffe-SSD with GPU (CUDA) installed in your Ubuntu machine you can skip the following steps and go directly to File Structure in section 5.**

We use Docker to build the Caffe-ssd environment. However, you will need to install Docker with GPU support (Cuda) before you can build the docker image. You can find a helpful procedure to install Docker/Cuda on Ubuntu-18.04 OS [here](https://confluencecommercial.flir.com/display/IISRT/Installation+guide+for+Docker-ce%2C+Cuda+and+Nvidia-drivers).

We tested the environment with the following host machine setup:
- Ubuntu 18.04.
- Cuda 10.0/Cudnn 7.
- Docker-ce 19.03.12.
- Nvidia GeForce GTX 1080 GPU.

This command will pull the docker caffe-ssd image `asigiuk/caffe-ssd_devel:latest` and run a docker container with the following environment settings:
  - Caffe-SSD
  - Opencv-3.4.3
  - CUDA-8/CUDNN-7 GPU drive support.

```bash
cd docker/
docker run --gpus all --rm -it --name caffe-env-1  -e DISPLAY=${DISPLAY}  --net=host  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  -v /dev:/dev -v path/to/host_target_directory:/home/docker asigiuk/caffe-ssd_devel:latest
```

Important Notes:
1. Execute the `docker run` command under the `docker/` directory in this repository. You should find the following files inside this directory: caffe -ssd.dockerfile and Makefile.config.
2. Modify `-v path/to/host_target_directory:/home/docker` in the above command and replace `path/to/host_target_directory` with your host machine target directory path. This will mount specified your target host directory to the docker container home directory `/home/docker`.
3. Confirm that the container has access to the training images by saving the images under the your specified target host directory. Alternatively, you can add another volume mount argument to the `docker run` command .
4. The docker `-v` or `--volume` flag is used to mount a target directory in your host machine (i.e. `path/to/host_target_directory`) to the docker container directory (i.e. `/home/docker`) . You can find more information regarding the `docker run` command [here](https://docs.docker.com/engine/reference/commandline/run/#mount-volume--v---read-only)  

Confirm correct caffe-ssd build by running the following command inside the caffe-ssd docker container environment.
```bash
make /opt/caffe/runtest
``` -->

## File Structure
This provides an overview of your file structure expectations. Place your files in the following fashion to work with the scripts provided in this tutorial.

Below is the expected dataset structure.
```
DATASETS     
├── DATASET
|  ├── DATASET_IDENTIFIER
|  │   ├── Images
|  │   │  ├── image1.jpg
|  │   │  ├── image2.jpg
|  │   │  ├── image3.jpg
|  │   │   :
|  │   │   :
|  │   │   :   
|  │   │  └── imageN.jpg
|  │   └── Annotations
|  │      ├── image1.xml
|  │      ├── image2.xml
|  │      ├── image3.xml
|  │       :
|  │       :
|  │       :   
|  │      └── imageN.xml
|  └── augmented
|      ├── Images
|      │  ├── image1.jpg
|      │  ├── image2.jpg
|      │  ├── image3.jpg
|      │   :
|      │   :
|      │   :   
|      │  └── imageN.jpg
|      ├── Annotations
|      │  ├── image1.xml
|      │  ├── image2.xml
|      │  ├── image3.xml
|      │   :
|      │   :
|      │   :   
|      │  └── imageN.xml
|      ├── ImageSets
|      │  └── Main
|      │     ├── trainval.txt
|      │     └── test.txt
|      └── lmdb
|          ├── trainval_lmdb
|          │   ├── data.mdb
|          │   └── lock.mdb
|          └── test_lmdb
|               ├── data.mdb
|               └── lock.mdb
└── DATASET2
    ├── DATASET_IDENTIFIER
        ├── Images
        └── Annotations                
```

Below is the structure of the data preparation and training scripts.
```
IIS_Object_Detection
├── README.md
├── proj.config
├── PrepareForTraining.py
├── Config.py
├── train.py
├── template
│  └── MobileNet-SSD
│      └── mobilenet_iter_73000.caffemodel
└── PROJECT_NAME (Your project scripts and logs will be created and saved under this folder)
   ├── MobileNetSSD_train.prototxt
   ├── MobileNetSSD_test.prototxt
   ├── MobileNetSSD_deploy.prototxt
   ├── train.prototxt
   ├── deploy.prototxt
   ├── solver_train.prototxt
   └── solver_test.prototxt
```

## Train Your Own Model
### Download Training Scripts
Download MobileNet-SSD training scripts, including the pretrained model weights, from the FLIR repo here.  These scripts are modified based on the original [MobileNet-SSD repo](https://github.com/chuanqi305/MobileNet-SSD).

```bash
git clone https://github.com/FLIR/IIS_Object_Detection.git
# todo: change when updating external
# this repo
```

A 'MobileNet-SSD' folder is included under 'IIS_Object_Detection/template/'

### Train
Lightning Memory-Mapped Database (LMDB) is a software library that provides a high-performance embedded transactional database in the form of a key-value store.

### Define configuration file in "IIS_Object_Detection/project.config"
First, `cd IIS_Object_Detection`. Modify proj.config file for your project.  

docker/linux
```json
{
    "ABSOLUTE_DATASETS_PATH": "/home/docker/dataset",
    "DATASET_FODLER": "dataset_folder",
    "DATASET_IDENTIFIER": "dataset_identifier",
    "IMAGE_FOLDER_NAME": "Images",
    "ANNOTAION_FOLDERNAME": "Annotations",
    "IMAGE_EXTENSION": "png",
    "TEST_CLASSES": ["QR"],
    "CAFFE_EXECUTION_COUNT":0,
    "PROJECT_NAME": "some_project_name",
    "TEST_SET_PERCENTAGE": 10,
    "ABSOLUTE_PATH_NETWORK": "/home/docker/caffe_ssd/template/MobileNet-SSD",
    "PRETRAINED_NETWORK_FILE": "mobilenet_iter_73000.caffemodel",
    "ABSOLUTE_OUTPUT_PROJECT_PATH": "/home/docker/dorker_proj/",
    "PATH_TO_CONVERT_ANNOSET_DOT_EXE": "/opt/caffe/build/tools/convert_annoset",
    "PATH_TO_GET_IMAGE_SIZE_DOT_EXE": "/opt/caffe/build/tools/get_image_size",
    "PATH_TO_CAFFE_DOT_EXE": "/opt/caffe/build/tools/caffe",
    "CONTINUE_TRAINING": false
}
```

Windows
```json
{
    "ABSOLUTE_DATASETS_PATH": "C:\\path\\to\\dataset",
    "DATASET_FODLER": "dataset",
    "DATASET_IDENTIFIER": "dataset_id",
    "IMAGE_FOLDER_NAME": "Images",
    "ANNOTAION_FOLDERNAME": "Annotations",
    "IMAGE_EXTENSION": "png",
    "TEST_CLASSES": ["QR"],
    "PROJECT_NAME": "some_project_1",
    "TEST_SET_PERCENTAGE": 10,
    "ABSOLUTE_PATH_NETWORK": "C:\\path\\to\\template\\MobileNet-SSD",
    "PRETRAINED_NETWORK_FILE": "mobilenet_iter_73000.caffemodel",
    "ABSOLUTE_OUTPUT_PROJECT_PATH": "here",
    "PATH_TO_CONVERT_ANNOSET_DOT_EXE": "C:\\path\\to\\MobileNet-SSD-windows\\scripts\\build\\tools\\Release\\convert_annoset.exe",
    "PATH_TO_GET_IMAGE_SIZE_DOT_EXE": "C:\\path\\to\\MobileNet-SSD-windows\\scripts\\build\\tools\\Release\\get_image_size.exe",
    "PATH_TO_CAFFE_DOT_EXE": "C:\\path\\to\\MobileNet-SSD-windows\\scripts\\build\\tools\\Release\\caffe.exe",
    "CONTINUE_TRAINING": false,
    "CAFFE_EXECUTION_COUNT":0,
}
```
<!-- #### Generate LMDB files & train your model -->
### Generate your project folder
```bash
python3 PrepareForTraining.py proj.config
```
The script 'PrepareForTraining.py' generates the project folder and copies the training script along with its configurations as well as prepares the data specified to for trianing.
<!-- generates LMDB files and calls 'train.sh' for training your own model. The training output files (*_iter_*.caffemodel, *_iter_*.solverstate) are saved under "IIS_Object_Detection/${script_folder_name}/snapshot" directory.-->
### Training
** The training can be terminated early if the loss is at a satisfactory level and has stopped decreasing. The latest model weight will be saved. **
Replace $ABSOLUTE_OUTPUT_PROJECT_PATH/$PROJECT_NAME with the path specified in the configuration
```bash
cd $ABSOLUTE_OUTPUT_PROJECT_PATH/$PROJECT_NAME
python3.5 train.py $PROJECT_NAME.config
```
### Resuming Training

change CONTINUE_TRAINING to true in your configuration file or use the newer config file
```json
"CONTINUE_TRAINING": true
```
```bash
cd $ABSOLUTE_OUTPUT_PROJECT_PATH/$PROJECT_NAME
python3.5 train.py $PROJECT_NAME-$CAFFE_EXECUTION_COUNT.config
```
### Training Notes

- If everything works we can start making sure our training gives good/better results
- If running a GPU with more graphics memory increase the batch size in MobileNetSSD_train.prototxt to improve training speed
- Edit solver_train.prototxt's max_iter to something large to increase the number of training iterations

### Testing Training
Test your trained model and evaluate the result.
```bash
cd IIS_Object_Detection/${PROJECT_NAME}
path\to\your\caffe\build\MobileNet-SSD-windows\scripts\build\tools\Release\caffe.exe train \
-solver="solver_test.prototxt" \
-weights=path\to\your\latest\snapshot \
-gpu 0

```


### Deploy
Trained model saved under `snapshot/` directory..

After we trained our model, we use NeuroUtility to convert the model to Firefly DL format, and upload it to a Firefly DL camera.

### Inference on camera
Labelfile will be generated with the contents based on the classes in your dataset.
```
background
class1
class2
class3
```
The label file can be found at: "IIS_Object_Detection/template/\<label_file\>.txt".

Right click SpinView, select "Configure Inference Label", and Browse to the label file, click "Apply". Now, enable inference, and stream the camera. You should be able to have Firefly-DL camera localizing QR code now.

### AppendingNewData (Highly Experimental)
If you wish to add additional data/images or classes. We will need to put it into another folder similar to how the first set was setup then create a new config (or copy the old one) and point the image and annotations folder to the new location. Then run

```bash
python AppendNewData.py <old_config> <new_config>
```

this will remove the old lmdb and add new images to the trainval and test set then rebuild the lmdb and the prototxt files to accomodate the change in the number of classes and images.

this is Highly Experimental atm and should only be used in a very specific way 
