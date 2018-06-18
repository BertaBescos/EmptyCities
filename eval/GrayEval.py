import os
import sys
import numpy as np
import skimage.io
import cv2
import argparse

argparser = argparse.ArgumentParser(description='Eval')
argparser.add_argument(
'folder',
type=str,
help = 'Missing folder path')

args = argparser.parse_args()

list = [str(50),str(100),str(150),'latest']

for idl in range(0,4):

	OUTPUT_DIR = args.folder + '/' + list[idl] + '_net_G_test/images/output/'
	TARGET_DIR = args.folder + '/' + list[idl]+ '_net_G_test/images/target/'
	INPUT_DIR = args.folder + '/' + list[idl]+ '_net_G_test/images/input/'
	MASK_DIR = '/home/bescosb/CARLA_0.8.2/dataset/MaskDyn/test/'

	print(list[idl])

	file_names = sorted(next(os.walk(OUTPUT_DIR))[2])

	id = 0
	id2 = 0
	
	vL1normGRAY = np.zeros((len(file_names),1))
	vL2normGRAY = np.zeros((len(file_names),1))

	vmeanL1normGRAY = np.zeros((len(file_names),1))
	vmeanL2normGRAY = np.zeros((len(file_names),1))

	vL1normGRAYMask = np.zeros((len(file_names),1))
	vL2normGRAYMask = np.zeros((len(file_names),1))

	vmeanL1normGRAYMask = np.zeros((len(file_names),1))
	vmeanL2normGRAYMask = np.zeros((len(file_names),1))
		
	for name in file_names:

		output = skimage.io.imread(os.path.join(OUTPUT_DIR,name))
		target = skimage.io.imread(os.path.join(TARGET_DIR,name))
		input = skimage.io.imread(os.path.join(INPUT_DIR,name))
		mask = skimage.io.imread(os.path.join(MASK_DIR,name))
		
		mask = cv2.resize(mask,(286,286))
		oW = 256
		oH = 256
		iW = mask.shape[1]
		iH = mask.shape[0]
		h1 = iH - oH
		w1 = iW - oW
		mask = mask[h1 : h1 + oH, w1 : w1 + oW]

		L1normGRAY = cv2.norm(output,target,cv2.NORM_L1)

		vL1normGRAY[id] = L1normGRAY

		L2normGRAY = cv2.norm(output,target,cv2.NORM_L2)

		vL2normGRAY[id] = L2normGRAY

		nPixels = output.size

		meanL2normGRAY = L2normGRAY/nPixels
		meanL1normGRAY = L1normGRAY/nPixels

		vmeanL1normGRAY[id] = meanL1normGRAY
		vmeanL2normGRAY[id] = meanL2normGRAY
	
		aux = np.zeros((256,256))
		aux[mask[:,:,0] == 255] = 1
		nPixelsMask = np.sum(aux)
		
		L1normGRAYMask = cv2.norm(output,target,cv2.NORM_L1,np.uint8(aux))

		vL1normGRAYMask[id] = L1normGRAYMask

		L2normGRAYMask = cv2.norm(output,target,cv2.NORM_L2,np.uint8(aux))

		vL2normGRAYMask[id] = L2normGRAYMask	
	
		if nPixelsMask != 0:

			meanL2normGRAYMask = L2normGRAYMask/nPixelsMask
			meanL1normGRAYMask = L1normGRAYMask/nPixelsMask

			vmeanL1normGRAYMask[id2] = meanL1normGRAYMask
			vmeanL2normGRAYMask[id2] = meanL2normGRAYMask

			id2 = id2 + 1

		id = id + 1
		
	L1normGRAY = np.median(vL1normGRAY)
	L2normGRAY = np.median(vL2normGRAY)

	meanL1normGRAY = np.median(vmeanL1normGRAY)
	meanL2normGRAY = np.median(vmeanL2normGRAY)

	L1normGRAYMask = np.median(vL1normGRAYMask)
	L2normGRAYMask = np.median(vL2normGRAYMask)

	meanL1normGRAYMask = np.median(vmeanL1normGRAYMask)
	meanL2normGRAYMask = np.median(vmeanL2normGRAYMask)

	file = open(args.folder + '/evals/' + list[idl]+".txt","a")
	file.write("%10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n" % (L1normGRAY, L2normGRAY, meanL1normGRAY, meanL2normGRAY, L1normGRAYMask, L2normGRAYMask, meanL1normGRAYMask, meanL2normGRAYMask))
	file.close()
