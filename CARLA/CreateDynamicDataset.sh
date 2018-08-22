#!/bin/bash

for w in {0..3}
do
	for id in {35..150..50}
	do
		python PythonClient/client_example.py $w $id -i -a
		
		src="_out/episode_0000/*"
		destWId="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/Dynamic/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory.txt"
		mkdir $destWId
		mkdir $dest
		mv $src $dest 
		mv $ctrl $dest
		mv $trj $dest
	done	
done


for w in {4..7}
do
	for id in {55..150..50}
	do
		python PythonClient/client_example.py $w $id -i -a
		
		src="_out/episode_0000/*"
		destWId="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/Dynamic/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory.txt"
		mkdir $destWId
		mkdir $dest
		mv $src $dest 
		mv $ctrl $dest
		mv $trj $dest
	done
done

for w in {8..11}
do
	for id in {5..150..50}
	do
		python PythonClient/client_example.py $w $id -i -a
		
		src="_out/episode_0000/*"
		destWId="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/Dynamic/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory.txt"
		mkdir $destWId
		mkdir $dest
		mv $src $dest 
		mv $ctrl $dest
		mv $trj $dest
	done
done

for w in {12..14}
do
	for id in {25..150..50}
	do
		python PythonClient/client_example.py $w $id -i -a
		
		src="_out/episode_0000/*"
		destWId="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/"
		dest="/home/bescosb/CARLA_0.8.2/dataset/Town02_val/"$w"_"$id"/Dynamic/"
		ctrl="/home/bescosb/CARLA_0.8.2/Control.txt"
		trj="/home/bescosb/CARLA_0.8.2/Trajectory.txt"
		mkdir $destWId
		mkdir $dest
		mv $src $dest 
		mv $ctrl $dest
		mv $trj $dest
	done
done

