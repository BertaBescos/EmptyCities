#!/bin/bash

dir="$1"
echo $dir

for w in {0..14}; do
	_w=`printf %02d $w`
	list=$(ls -d $dir"Town02Test/"$_w"_"*)
	list_length=$(ls -d $dir"Town02Test/"$_w"_"* | wc -l)
	arr=($list)
	for id in $(seq 0 $(($list_length - 1))); do
		dir_stat=${arr[$id]}"/Static"
		echo $dir_stat
		if [ ! -d "$dir_stat" ]; then
			mkdir $dir_stat
		fi
		ctrl=${arr[$id]}"/Dynamic/Control.txt"
		trj=${arr[$id]}"/Dynamic/Trajectory.txt"

		player_id=$(cut -d "_" -f 2 <<< "${arr[$id]}")

		python PythonClient/client_example_read.py $w $player_id $ctrl $trj -i -a

		dyn_src=${arr[$id]}"/Dynamic"
		src="./_out_s/episode_0000/*"
		dest=${arr[$id]}"/Static"
		ctrl="./Control_s.txt"
		trj="./Trajectory_s.txt"

		mv $src $dest 
		mv $ctrl $dest
		mv $trj $dest

		nImStatic=$(ls $dest"RGB/" | wc -l)
		nImDynamic=$(ls $dyn_src"RGB/" | wc -l)
		dynFilesRGB=($dyn_src"RGB"/*)
		dynFilesRGB_back=($dyn_src"RGB_back"/*)
		dynFilesDepth=($dyn_src"Depth"/*)
		dynFilesDepth_back=($dyn_src"Depth_back"/*)
		dynFilesSemanticSegmentation=($dyn_src"SemanticSegmentation"/*)
		dynFilesSemanticSegmentation_back=($dyn_src"SemanticSegmentation_back"/*)
		if [ "$nImStatic" -ne "$nImDynamic" ]
		then
			for file in $(find $dyn_src"RGB/" $dest"RGB/" $dest"RGB/" -printf '%P\n' | sort | uniq -u);
			do
				rm $dyn_src"RGB/"$file
				rm $dyn_src"RGB_back/"$file
				rm $dyn_src"Depth/"$file
				rm $dyn_src"Depth_back/"$file
				rm $dyn_src"SemanticSegmentation/"$file
				rm $dyn_src"SemanticSegmentation_back/"$file
			done
		fi
	done
done
