#!/bin/bash

# [1] Define your configureations
#CAFFE_ROOT=/opt/caffe  # Do not need this for the new docker image 
DATA_DIR="$HOME/dixu/Datasets/QRData/test" #augmentated
IMAGE_FORMAT='.png'     # Only one format is allowed for all images
TEST_SET_PERCENTAGE=15  # percentage of all images that goes to test set, the rest will go to trainval set
script_folder_name="qr" # used in create_data.sh 


# [2] Copy and modify labelmap.prototxt 
FILE=data/${script_folder_name}/labelmap.prototxt
if test -f "$FILE"; then
    echo "$FILE exists."
else 
	echo "$FILE does not exist."
	cp data/my_template/labelmap.prototxt data/${script_folder_name}
fi
# Need to modify the labelmap.prototxt before running the rest of the scripts
# ===============================================================

# [3] Some processing to prepare for the rest of the scripts
data_root_dir=`dirname "$DATA_DIR"`   #data_root_dir=$HOME/dixu/Datasets/QRData
dataset_name=`basename "$DATA_DIR"`   #dataset_name="augmented"
echo ''
echo 'data_root_dir:' $data_root_dir
echo 'dataset_name:' $dataset_name


# [4] run data_partition.sh
echo 'Running data_partition ...'
./data/my_template/data_partition.sh ${DATA_DIR} ${TEST_SET_PERCENTAGE}


# [5] run create_list.sh
echo 'Running create_list ...'
echo 'mkdir: ' data/${script_folder_name}
mkdir -p data/${script_folder_name}
./data/my_template/create_list.sh $data_root_dir $dataset_name $IMAGE_FORMAT $script_folder_name


# [6] run create_data.sh
echo 'Running create_list ...'
./data/my_template/create_data.sh ${data_root_dir} ${dataset_name} ${script_folder_name}
