#!/bin/sh
CAFFE_ROOT="$HOME/dixu/Projects/caffe_root"
if ! test -f MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
$CAFFE_ROOT/build/tools/caffe train -solver="solver_train.prototxt" \
-weights="../MobileNet-SSD/mobilenet_iter_73000.caffemodel" \
-gpu 0 
