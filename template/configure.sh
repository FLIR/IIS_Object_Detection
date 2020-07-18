#CAFFE_ROOT=/opt/caffe  # Do not need this for the new docker image 
DATA_DIR="$HOME/dixu/Datasets/QRData/augmented"
IMAGE_FORMAT='.png'     # Only one format is allowed for all images
TEST_SET_PERCENTAGE=15  # percentage of all images that goes to test set, the rest will go to trainval set
script_folder_name="qr" # used in create_data.sh 
data_root_dir=`dirname "$DATA_DIR"`   #data_root_dir=$HOME/dixu/Datasets/QRData
dataset_name=`basename "$DATA_DIR"`   #dataset_name="augmented"
echo 'data_root_dir:' $data_root_dir
echo 'dataset_name:' $dataset_name
