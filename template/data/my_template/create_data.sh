cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..
#CAFFE_ROOT=$HOME/dixu/Projects/caffe_root  # This has compiled caffe library. Is part of docker configuration     

#cd $root_dir

redo=1
#data_root_dir="$HOME/dixu/Datasets/QRData"
#dataset_name="augmented"               # lmdb files will be generated under $data_root_dir/$dataset_name/
#script_folder_name="qr"
mapfile="$cur_dir/labelmap.prototxt"   #"$root_dir/data/$dataset_name/labelmap.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python3 $CAFFE_ROOT/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $cur_dir/$subset.txt $data_root_dir/$dataset_name/$db/$subset"_"$db ../../examples/$script_folder_name
done
