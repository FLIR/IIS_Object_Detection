#!/bin/sh
#CAFFE_ROOT="$HOME/dixu/Projects/caffe_root"
CAFFE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"../template/caffe_root && pwd)"
#CAFFE_DIR="../template"
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
$CAFFE_ROOT/build/tools/caffe train -solver="solver_test.prototxt" \
#${CAFFE_DIR}/caffe train -solver="solver_test.prototxt" \
--weights=$latest \
-gpu 0
