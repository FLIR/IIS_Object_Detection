#!/bin/bash

data_root_dir=$1 
dataset_name=$2 
IMAGE_FORMAT=$3 
script_folder_name=$4

CAFFE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"../caffe_root && pwd)"
sub_dir=ImageSets/Main
#bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../${script_folder_name} && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi

  echo "Create list for $dataset_name $dataset..."
  dataset_file=$data_root_dir/$dataset_name/$sub_dir/$dataset.txt

  img_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_file
  sed -i "s/^/$dataset_name\/Images\//g" $img_file
  sed -i "s/$/$IMAGE_FORMAT/g" $img_file  

  label_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_file
  sed -i "s/^/$dataset_name\/Annotations\//g" $label_file
  sed -i "s/$/.xml/g" $label_file

  paste -d' ' $img_file $label_file >> $dst_file

  rm -f $label_file
  rm -f $img_file


  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    #$bash_dir/../../build/tools/get_image_size $data_root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
    $CAFFE_ROOT/build/tools/get_image_size $data_root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
