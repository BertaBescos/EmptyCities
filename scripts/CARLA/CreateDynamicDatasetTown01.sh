#!/bin/bash
# Copyright (C) 2018 Berta Bescos 
# <bbescos at unizar dot es> (University of Zaragoza)
#

dir="$1"
echo $dir
if [ ! -d "$dir" ]; then
	mkdir $dir
fi
dir_train=$dir"Town01Train"
if [ ! -d "$dir_train" ]; then
	mkdir $dir_train
fi
dir_val=$dir"Town01Val"
if [ ! -d "$dir_val" ]; then
	mkdir $dir_val
fi

for w in {0..14}; do
	id1=$((1 + RANDOM % 30))
	id2=$((31 + RANDOM % 30))
	id3=$((61 + RANDOM % 30))
	id4=$((91 + RANDOM % 30))
	id5=$((121 + RANDOM % 30))

	echo $id1 $id2 $id3 $id4 $id5

	for id in $id1 $id2 $id3 $id4 $id5; do
		python PythonClient/client_example_write.py $w $id -i -a
		src="./_out/episode_0000/*"
		_w=`printf %02d $w`
		_id=`printf %03d $id`
		destWId=$dir_train"/"$_w"_"$_id"/"
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

for w in {0..14}; do
	id1=$((1 + RANDOM % 150))
	for id in $id1; do
		python PythonClient/client_example_write.py $w $id -i -a
		src="./_out/episode_0000/*"
		_w=`printf %02d $w`
		_id=`printf %03d $id`
		destWId=$dir_val"/"$_w"_"$_id"/"
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