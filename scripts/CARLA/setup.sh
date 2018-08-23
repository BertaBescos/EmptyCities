#!/bin/bash

src="$1"
dest="$2"
if [ ! -d "$dest" ]; then
	mkdir $dest
fi
dest_A=$dest"A/"
if [ ! -d "$dest_A" ]; then
	mkdir $dest_A
fi
dest_B=$dest"B/"
if [ ! -d "$dest_B" ]; then
	mkdir $dest_B
fi
dest_C=$dest"C/"
if [ ! -d "$dest_C" ]; then
	mkdir $dest_C
fi

train_folders=($src"Town01Train/*/")
echo $train_folders
imOut_ini=0
for folder in "${train_folders[@]}"; do
	ims=($folder"Dynamic/RGB"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_A"train/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/RGB_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_A"train/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Static/RGB"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_B"train/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Static/RGB_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_B"train/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/SemanticSegmentation"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_C"train/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/SemanticSegmentation_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_C"train/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	imOut_ini=$imOut
done

test_folders=($src"Town02Test/*/")
echo $test_folders
for folder in "${test_folders[@]}"; do
	ims=($folder"Dynamic/RGB"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_A"test/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/RGB_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_A"test/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Static/RGB"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_B"test/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Static/RGB_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_B"test/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/SemanticSegmentation"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_C"test/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/SemanticSegmentation_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_C"test/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	imOut_ini=$imOut
done

val_folders=($src"Town01Val/*/")
echo $val_folders
for folder in "${val_folders[@]}"; do
	ims=($folder"Dynamic/RGB"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_A"val/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/RGB_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_A"val/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Static/RGB"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_B"val/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Static/RGB_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_B"val/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/SemanticSegmentation"/*)
	imOut=$imOut_ini
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_C"val/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	ims=($folder"Dynamic/SemanticSegmentation_back"/*)
	for im in "${ims[@]}"; do
		_imOut=`printf %06d $imOut`
		cp $im $dest_C"val/"$_imOut".png"
		imOut=$(($imOut + 1))
	done
	imOut_ini=$imOut
done
