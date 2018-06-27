import os
import sys
import numpy as np
import skimage.io
import cv2
import argparse

argparser = argparse.ArgumentParser(description='Eval')

argparser.add_argument(
'--output_folder',
type=str,
help = 'Missing output folder path')

argparser.add_argument(
'--target_folder',
type=str,
help = 'Missing target folder path')

argparser.add_argument(
'--mask_folder',
type=str,
help = 'Missing mask folder path')

args = argparser.parse_args()

file_names = sorted(next(os.walk(args.output_folder))[2])
accumL1 = 0
accummaskL1 = 0
accumnomaskL1 = 0
n = 0

for name in file_names:

	output = cv2.imread(os.path.join(args.output_folder,name),0)
	output = cv2.resize(output,(286,286))
	target = cv2.imread(os.path.join(args.target_folder,name),0)
	target = cv2.resize(target,(286,286))
	mask = cv2.imread(os.path.join(args.mask_folder,name))
	mask = cv2.resize(mask,(286,286))
	
	bin_mask = np.zeros((mask.shape[0],mask.shape[1]))
	bin_nomask = np.ones((mask.shape[0],mask.shape[1]))
	bin_mask[mask[:,:,0] == 255] = 1
	bin_nomask[mask[:,:,0] == 255] = 0
	nPixels = np.sum(bin_mask)
	nNoPixels = np.sum(bin_nomask)	

	if nPixels > 0:
		L1 = cv2.norm(output,target,cv2.NORM_L1)
		L1 = L1/286/286
		L1 = L1/255*100
		L1mask = cv2.norm(output,target,cv2.NORM_L1,np.uint8(bin_mask))
		L1mask = L1mask/nPixels				
		L1mask = L1mask/255*100
		L1nomask = cv2.norm(output,target,cv2.NORM_L1,np.uint8(bin_nomask))
		L1nomask = L1nomask/nNoPixels				
		L1nomask = L1nomask/255*100
		accumL1 = accumL1 + L1
		accummaskL1 = accummaskL1 + L1mask
		accumnomaskL1 = accumnomaskL1 + L1nomask
		n = n + 1

meanL1 = accumL1/n
meanmaskL1 = accummaskL1/n
meannomaskL1 = accumnomaskL1/n
print('meanL1: ', meanL1)
print('meanL1 mask: ', meanmaskL1)
print('meanL1 nomask: ', meannomaskL1)
