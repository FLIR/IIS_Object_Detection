#!/bin/bash
. ../project.config

## [1] DATA
# [1a] Generate labelmap.prototxt file
echo 'mkdir: ' data/${script_folder_name}
mkdir -p data/${script_folder_name}
FILE=data/${script_folder_name}/labelmap.prototxt
if test -f "$FILE"; then
    rm $FILE
fi
touch $FILE

# Write backgound class
echo "item {
  name: \"none_of_the_above\"
  label: 0
  display_name: \"background\"
}" >> $FILE
# Write the rest of classes
ind_class=0
IFS=',' read -ra ADDR <<< "$CLASSES"
for i in "${ADDR[@]}"; do
	ind_class=$((ind_class+1))
	echo "$i"
	echo "item {
  name: \"$i\"
  label: $ind_class
  display_name: \"$i\"
}" >> $FILE
done

# ===============================================================

# [1b] Some processing to prepare for the rest of the scripts
data_root_dir=`dirname "$DATA_DIR"`   #data_root_dir=$HOME/dixu/Datasets/QRData
dataset_name=`basename "$DATA_DIR"`   #dataset_name="augmented"
echo ''
echo 'data_root_dir:' $data_root_dir
echo 'dataset_name:' $dataset_name


# [1c] run data_partition.sh
echo 'Running data_partition ...'
./data/my_template/data_partition.sh ${DATA_DIR} ${TEST_SET_PERCENTAGE}


# [1d] run create_list.sh
echo 'Running create_list ...'
./data/my_template/create_list.sh $data_root_dir $dataset_name $IMAGE_FORMAT $script_folder_name


# [1e] run create_data.sh
echo 'Running create_list ...'
./data/my_template/create_data.sh ${data_root_dir} ${dataset_name} ${script_folder_name}


## [2] TRAIN 


# [2a] Use gen_model.sh to generate your own prototxt files for training and deployment
cd MobileNet-SSD
./gen_model.sh $CLASSNUM 

# [2b] Create a 'qr' folder
ProjectFolder="../../"${script_folder_name} #"caffe_ssd/qr"
mkdir ${ProjectFolder}

# [2c] Copy files to the 'ProjectFolder' 
cp example/* ${ProjectFolder}
# cp mobilenet_iter_73000.caffemodel ${ProjectFolder} 
cp *.prototxt ${ProjectFolder}
cp train.sh ${ProjectFolder}
cp test.sh ${ProjectFolder}


# [2d] Create symlinks to lmdb data:
cd ${ProjectFolder}
ln -s ${DATA_DIR}/lmdb/trainval_lmdb trainval_lmdb
ln -s ${DATA_DIR}/lmdb/test_lmdb test_lmdb


# [2e] Train
./train.sh
