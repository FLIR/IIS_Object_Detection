#!/bin/bash
#DATA_DIR="$HOME/dixu/Datasets/QRData/augmented"
DATA_DIR=$1
FILES="${DATA_DIR}/Images/*${IMAGE_FORMAT}"  #"${DATA_DIR}/Images/*.png"
TEST_SET_PERCENTAGE=$2  # percentage of all images that goes to test set, the rest will go to trainval set

# create folder if it does not exist
DATA_PARTATION_FILE_DIR="${DATA_DIR}/ImageSets/Main"
mkdir -p $DATA_PARTATION_FILE_DIR

# create files if do not exist
FILE=${DATA_PARTATION_FILE_DIR}/trainval.txt
if test -f "$FILE"; then
    echo "$FILE exists. Remove it and recreate a new file."
    rm ${DATA_PARTATION_FILE_DIR}/trainval.txt
    rm ${DATA_PARTATION_FILE_DIR}/test.txt
else 
    echo "$FILE does not exist."
fi
touch ${DATA_PARTATION_FILE_DIR}/trainval.txt
touch ${DATA_PARTATION_FILE_DIR}/test.txt

RANDOM=123  # seed for the random numbers

# partition files into trainval & test sets
for f in $FILES
do
    # parse path, file name, and file extension
    path=${f%/*}
    filename_w_ext=${f##*/}
    ext=${filename_w_ext##*.}
    filename_wo_ext=${filename_w_ext%.*}

    echo "Processing ${filename_w_ext} ..."
    # generate a random number between 0 and 100, inclusive
    randn=$((RANDOM%101))
    if ((${randn} < ${TEST_SET_PERCENTAGE})); then
        # echo "added to test set"
        echo ${filename_wo_ext} >> ${DATA_PARTATION_FILE_DIR}/test.txt

    else
        # echo "added to trainval set"
        echo ${filename_wo_ext} >> ${DATA_PARTATION_FILE_DIR}/trainval.txt
    fi
done
