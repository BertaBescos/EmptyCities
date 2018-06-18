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

	vL1normRGB = np.zeros((len(file_names),1))
	vL1normGRAY = np.zeros((len(file_names),1))
	vL2normRGB = np.zeros((len(file_names),1))
	vL2normGRAY = np.zeros((len(file_names),1))

	vmeanL1normRGB = np.zeros((len(file_names),1))
	vmeanL1normGRAY = np.zeros((len(file_names),1))
	vmeanL2normRGB = np.zeros((len(file_names),1))
	vmeanL2normGRAY = np.zeros((len(file_names),1))

	vL1normRGBMask = np.zeros((len(file_names),1))
	vL1normGRAYMask = np.zeros((len(file_names),1))
	vL2normRGBMask = np.zeros((len(file_names),1))
	vL2normGRAYMask = np.zeros((len(file_names),1))

	vmeanL1normRGBMask = np.zeros((len(file_names),1))
	vmeanL1normGRAYMask = np.zeros((len(file_names),1))
	vmeanL2normRGBMask = np.zeros((len(file_names),1))
	vmeanL2normGRAYMask = np.zeros((len(file_names),1))

	for name in file_names:

		output = skimage.io.imread(os.path.join(OUTPUT_DIR,name))
		output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
		target = skimage.io.imread(os.path.join(TARGET_DIR,name))
		target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
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

		L1normRGB = cv2.norm(output,target,cv2.NORM_L1)
		L1normGRAY = cv2.norm(output_gray,target_gray,cv2.NORM_L1)

		vL1normRGB[id] = L1normRGB
		vL1normGRAY[id] = L1normGRAY

		L2normRGB = cv2.norm(output,target,cv2.NORM_L2)
		L2normGRAY = cv2.norm(output_gray,target_gray,cv2.NORM_L2)

		vL2normRGB[id] = L2normRGB
		vL2normGRAY[id] = L2normGRAY

		nPixels = output_gray.size

		meanL2normRGB = L2normRGB/nPixels
		meanL1normRGB = L1normRGB/nPixels
		meanL2normGRAY = L2normGRAY/nPixels
		meanL1normGRAY = L1normGRAY/nPixels
	
		vmeanL1normRGB[id] = meanL1normRGB
		vmeanL1normGRAY[id] = meanL1normGRAY
		vmeanL2normRGB[id] = meanL2normRGB
		vmeanL2normGRAY[id] = meanL2normGRAY

		aux = np.zeros((256,256))
		aux[mask[:,:,0] == 255] = 1
		nPixelsMask = np.sum(aux)
		
		L1normRGBMask = cv2.norm(output,target,cv2.NORM_L1,np.uint8(aux))
		L1normGRAYMask = cv2.norm(output_gray,target_gray,cv2.NORM_L1,np.uint8(aux))

		vL1normRGBMask[id] = L1normRGBMask
		vL1normGRAYMask[id] = L1normGRAYMask

		L2normRGBMask = cv2.norm(output,target,cv2.NORM_L2,np.uint8(aux))
		L2normGRAYMask = cv2.norm(output_gray,target_gray,cv2.NORM_L2,np.uint8(aux))

		vL2normRGBMask[id] = L2normRGBMask
		vL2normGRAYMask[id] = L2normGRAYMask

		if nPixelsMask != 0:
			meanL2normRGBMask = L2normRGBMask/nPixelsMask
			meanL1normRGBMask = L1normRGBMask/nPixelsMask
			meanL2normGRAYMask = L2normGRAYMask/nPixelsMask
			meanL1normGRAYMask = L1normGRAYMask/nPixelsMask

			vmeanL1normRGBMask[id2] = meanL1normRGBMask
			vmeanL1normGRAYMask[id2] = meanL1normGRAYMask
			vmeanL2normRGBMask[id2] = meanL2normRGBMask
			vmeanL2normGRAYMask[id2] = meanL2normGRAYMask

			id2 = id2 + 1

		id = id + 1

	L1normRGB = np.median(vL1normRGB)
	L1normGRAY = np.median(vL1normGRAY)
	L2normRGB = np.median(vL2normRGB)
	L2normGRAY = np.median(vL2normGRAY)

	meanL1normRGB = np.median(vmeanL1normRGB)
	meanL1normGRAY = np.median(vmeanL1normGRAY)
	meanL2normRGB = np.median(vmeanL2normRGB)
	meanL2normGRAY = np.median(vmeanL2normGRAY)

	L1normRGBMask = np.median(vL1normRGBMask)
	L1normGRAYMask = np.median(vL1normGRAYMask)
	L2normRGBMask = np.median(vL2normRGBMask)
	L2normGRAYMask = np.median(vL2normGRAYMask)

	meanL1normRGBMask = np.median(vmeanL1normRGBMask)
	meanL1normGRAYMask = np.median(vmeanL1normGRAYMask)
	meanL2normRGBMask = np.median(vmeanL2normRGBMask)
	meanL2normGRAYMask = np.median(vmeanL2normGRAYMask)

	file = open(args.folder + '/evals/' + list[idl]+".txt","a")
	file.write("%10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n %10.10f \n" % (L1normRGB, L1normGRAY, L2normRGB, L2normGRAY, meanL1normRGB, meanL1normGRAY, meanL2normRGB, meanL2normGRAY, L1normRGBMask, L1normGRAYMask, L2normRGBMask, L2normGRAYMask, meanL1normRGBMask, meanL1normGRAYMask, meanL2normRGBMask, meanL2normGRAYMask))
	file.close()

