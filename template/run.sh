#!/bin/bash
. ../project.config

## DATA
# [2] Copy and modify labelmap.prototxt 
echo 'mkdir: ' data/${script_folder_name}
mkdir -p data/${script_folder_name}
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
./data/my_template/create_list.sh $data_root_dir $dataset_name $IMAGE_FORMAT $script_folder_name


# [6] run create_data.sh
echo 'Running create_list ...'
./data/my_template/create_data.sh ${data_root_dir} ${dataset_name} ${script_folder_name}


## TRAIN 


# Use gen_model.sh to generate your own prototxt files for training and deployment
cd MobileNet-SSD
./gen_model.sh $CLASSNUM 

# Create a 'qr' folder
ProjectFolder="../../"${script_folder_name} #"caffe_ssd/qr"
mkdir ${ProjectFolder}

# Copy files to the 'ProjectFolder' 
cp example/* ${ProjectFolder}
# cp mobilenet_iter_73000.caffemodel ${ProjectFolder} 
cp *.prototxt ${ProjectFolder}
cp train.sh ${ProjectFolder}
cp test.sh ${ProjectFolder}


cd ${ProjectFolder}
# Create symlinks to lmdb data:
ln -s ${DATA_DIR}/lmdb/trainval_lmdb trainval_lmdb
ln -s ${DATA_DIR}/lmdb/test_lmdb test_lmdb


./train.sh
