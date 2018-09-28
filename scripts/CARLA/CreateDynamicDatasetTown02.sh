#!/bin/bash
# Copyright (C) 2018 Berta Bescos 
# <bbescos at unizar dot es> (University of Zaragoza)
#

dir="$1"
echo $dir
if [ ! -d "$dir" ]; then
	mkdir $dir
fi
dir_test=$dir"Town02Test"
if [ ! -d "$dir_test" ]; then
	mkdir $dir_test
fi

for w in {0..14}; do
	id1=$((1 + RANDOM % 150))
	echo $id1
	for id in $id1; do
		python PythonClient/client_example_write.py $w $id -i -a
		src="./_out/episode_0000/*"
		_w=`printf %02d $w`
		_id=`printf %03d $id`
		destWId=$dir_test"/"$_w"_"$_id"/"
		dest=$destWId"Dynamic/"
		ctrl="./Control.txt"
		trj="./Trajectory.txt"
		mkdir $destWId
		mkdir $dest
		mv $src $dest
		mv $ctrl $dest
		mv $trj $dest
	done
done
