# Define your configureations
#CAFFE_ROOT=/opt/caffe  # Do not need this for the new docker image 
DATA_DIR="$HOME/dixu/Datasets/QRData/test" #augmentated
IMAGE_FORMAT='.png'     # Only one format is allowed for all images
TEST_SET_PERCENTAGE=15  # percentage of all images that goes to test set, the rest will go to trainval set
script_folder_name="qr" # used in create_data.sh 

# [1] Some processing to prepare for the rest of the scripts
data_root_dir=`dirname "$DATA_DIR"`   #data_root_dir=$HOME/dixu/Datasets/QRData
dataset_name=`basename "$DATA_DIR"`   #dataset_name="augmented"
echo ''
echo 'data_root_dir:' $data_root_dir
echo 'dataset_name:' $dataset_name

# [2] run data_partition.sh
#echo 'mkdir: ' data/${script_folder_name}
#mkdir -p data/${script_folder_name}
echo 'run data_partition'
./data/my_template/data_partition.sh ${DATA_DIR} ${TEST_SET_PERCENTAGE}

