#!/bin/bash

for w in {0..3}
do
	for id in {0..150..50}
	do
		ctrl="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Control.txt"
		traj="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Trajectory.txt"
		
		dyn_src="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/"

		python PythonClient/client_example_read.py $w $id $ctrl $traj -i -a
		
		src="_out_s/episode_0000/*"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Static/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control_s.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory_s.txt"
		mkdir $dest
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

for w in {4..7}
do
	for id in {10..150..50}
	do
		ctrl="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Control.txt"
		traj="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Trajectory.txt"
		
		dyn_src="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/"

		python PythonClient/client_example_read.py $w $id $ctrl $traj -i -a
		
		src="_out_s/episode_0000/*"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Static/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control_s.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory_s.txt"
		mkdir $dest
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

for w in {8..11}
do
	for id in {20..150..50}
	do
		ctrl="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Control.txt"
		traj="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Trajectory.txt"
		
		dyn_src="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/"

		python PythonClient/client_example_read.py $w $id $ctrl $traj -i -a
		
		src="_out_s/episode_0000/*"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Static/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control_s.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory_s.txt"
		mkdir $dest
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

for w in {12..14}
do
	for id in {30..150..50}
	do
		ctrl="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Control.txt"
		traj="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/Trajectory.txt"
		
		dyn_src="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Dynamic/"

		python PythonClient/client_example_read.py $w $id $ctrl $traj -i -a
		
		src="_out_s/episode_0000/*"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_test/"$w"_"$id"/Static/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control_s.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory_s.txt"
		mkdir $dest
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
