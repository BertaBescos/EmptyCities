import os
import sys
import numpy as np
import skimage.io
import cv2
import argparse
import matplotlib.pyplot as plt

RGB_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/RGB_Dyn_Stat/evals'
Gray_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/Gray_Dyn_Stat/evals'
RGBMask_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/RGB_MaskDyn_Stat/evals'
GrayMask_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/Gray_MaskDyn_Stat/evals'
GrayDiscMask_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/DiscGamma2/evals'
GrayDiscMask4_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/DiscGamma4/evals'
GrayDiscMask1_5_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/DiscGamma1.5/evals'
GrayDiscUpMask_Dyn_Stat = '/home/bescosb/pix2pix_0.1/results/CARLA/DiscGamma2AndUpsampling/evals'
GrayDiscMaskDataAug = '/home/bescosb/pix2pix_0.1/results/CARLA/DataAug/evals'


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(RGB_Dyn_Stat)) # ignore hidden folders

vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(RGB_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[5])
	meanL2normGRAY = float(lines[7])
	meanL1normGRAYMask = float(lines[13])
	meanL2normGRAYMask = float(lines[15])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1
	
plt.figure(1)
plt.subplot(221)
plt.plot([100,50,200,150],vmeanL1normGRAY,'ro',label = 'RGB')

plt.subplot(222)
plt.plot([100,50,200,150],vmeanL2normGRAY,'ro',label = 'RGB')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'ro',label = 'RGB')

plt.subplot(224)
plt.plot([100,50,200,150],vmeanL2normGRAYMask,'ro',label = 'RGB')


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(Gray_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(Gray_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(1)
plt.subplot(221)
plt.plot([100,50,200,150],vmeanL1normGRAY,'bo',label = 'Gray')

plt.subplot(222)
plt.plot([100,50,200,150],vmeanL2normGRAY,'bo',label = 'Gray')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'bo',label = 'Gray')

plt.subplot(224)
plt.plot([100,50,200,150],vmeanL2normGRAYMask,'bo',label = 'Gray')


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(RGBMask_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(RGBMask_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[5])
	meanL2normGRAY = float(lines[7])
	meanL1normGRAYMask = float(lines[13])
	meanL2normGRAYMask = float(lines[15])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(1)
plt.subplot(221)
L1norm = plt.plot([100,50,200,150],vmeanL1normGRAY,'mo',label = 'RGBMask')

plt.subplot(222)
L2norm = plt.plot([100,50,200,150],vmeanL2normGRAY,'mo',label = 'RGBMask')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'mo',label = 'RGBMask')

plt.subplot(224)
L2norm_mask = plt.plot([100,50,200,150],vmeanL2normGRAYMask,'mo',label = 'RGBMask')


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(GrayDiscMask_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(GrayDiscMask_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(1)
plt.subplot(221)
L1norm = plt.plot([100,50,200,150],vmeanL1normGRAY,'ko',label = 'GrayMask_Disc2.0')

plt.subplot(222)
L2norm = plt.plot([100,50,200,150],vmeanL2normGRAY,'ko',label = 'GrayMask_Disc2.0')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'ko',label = 'GrayMask_Disc2.0')

plt.subplot(224)
L2norm_mask = plt.plot([100,50,200,150],vmeanL2normGRAYMask,'ko',label = 'GrayMask_Disc2.0')


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(GrayDiscMaskDataAug)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(GrayDiscMaskDataAug + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(1)
plt.subplot(221)
L1norm = plt.plot([100,50,200,150],vmeanL1normGRAY,'k+',label = 'DataAug')

plt.subplot(222)
L2norm = plt.plot([100,50,200,150],vmeanL2normGRAY,'k+',label = 'DataAug')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'k+',label = 'DataAug')

plt.subplot(224)
L2norm_mask = plt.plot([100,50,200,150],vmeanL2normGRAYMask,'k+',label = 'DataAug')



# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(GrayMask_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(GrayMask_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(1)
plt.subplot(221)
plt.plot([100,50,200,150],vmeanL1normGRAY,'go',label = 'GrayMask')
plt.ylabel('L1norm')
plt.xlabel('epochs')
plt.axis([0,210,1.5,12.5])
plt.legend()

plt.subplot(222)
plt.plot([100,50,200,150],vmeanL2normGRAY,'go',label = 'GrayMask')
plt.axis([0,210,0.03,0.060])
plt.ylabel('L2norm')
plt.xlabel('epochs')
plt.axis([0,210,0.015,0.060])
plt.legend()

plt.subplot(223)
L1norm_mask = plt.plot([100,50,200,150],vmeanL1normGRAYMask,'go',label = 'GrayMask')
plt.ylabel('L1norm_mask')
plt.xlabel('epochs')
plt.axis([0,210,4,14])
plt.legend()

plt.subplot(224)
plt.plot([100,50,200,150],vmeanL2normGRAYMask,'go',label = 'GrayMask')
plt.axis([0,210,0.2,0.6])
plt.ylabel('L2norm_mask')
plt.xlabel('epochs')
plt.legend()

plt.show()



# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(GrayDiscMask_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(GrayDiscMask_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(2)
plt.subplot(221)
L1norm = plt.plot([100,50,200,150],vmeanL1normGRAY,'ko',label = 'GrayMask_Disc2.0')

plt.subplot(222)
L2norm = plt.plot([100,50,200,150],vmeanL2normGRAY,'ko',label = 'GrayMask_Disc2.0')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'ko',label = 'GrayMask_Disc2.0')

plt.subplot(224)
L2norm_mask = plt.plot([100,50,200,150],vmeanL2normGRAYMask,'ko',label = 'GrayMask_Disc2.0')


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(GrayDiscMask4_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(GrayDiscMask4_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(2)
plt.subplot(221)
L1norm = plt.plot([100,50,200,150],vmeanL1normGRAY,'r+',label = 'GrayMask_Disc4.0')

plt.subplot(222)
L2norm = plt.plot([100,50,200,150],vmeanL2normGRAY,'r+',label = 'GrayMask_Disc4.0')

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'r+',label = 'GrayMask_Disc4.0')

plt.subplot(224)
L2norm_mask = plt.plot([100,50,200,150],vmeanL2normGRAYMask,'r+',label = 'GrayMask_Disc4.0')


# Read files
splits = filter( lambda f: not f.startswith('.'), os.listdir(GrayDiscMask1_5_Dyn_Stat)) # ignore hidden folders
vmeanL1normGRAY = np.zeros((len(splits),1))
vmeanL2normGRAY = np.zeros((len(splits),1))
vmeanL1normGRAYMask = np.zeros((len(splits),1))
vmeanL2normGRAYMask = np.zeros((len(splits),1))
id = 0

for sp in splits:
	file = open(GrayDiscMask1_5_Dyn_Stat + '/' + sp,'r')
	lines = file.readlines()
	meanL1normGRAY = float(lines[2])
	meanL2normGRAY = float(lines[3])
	meanL1normGRAYMask = float(lines[6])
	meanL2normGRAYMask = float(lines[7])
	
	vmeanL1normGRAY[id] = meanL1normGRAY
	vmeanL2normGRAY[id] = meanL2normGRAY
	vmeanL1normGRAYMask[id] = meanL1normGRAYMask
	vmeanL2normGRAYMask[id] = meanL2normGRAYMask
	id = id + 1

plt.figure(2)
plt.subplot(221)
L1norm = plt.plot([100,50,200,150],vmeanL1normGRAY,'g+',label = 'GrayMask_Disc1.5')
plt.ylabel('L1norm_mask')
plt.xlabel('epochs')
plt.axis([0,210,1.5,5])
plt.legend()

plt.subplot(222)
L2norm = plt.plot([100,50,200,150],vmeanL2normGRAY,'g+',label = 'GrayMask_Disc1.5')
plt.ylabel('L1norm_mask')
plt.xlabel('epochs')
plt.axis([0,210,0,0.1])
plt.legend()

plt.subplot(223)
plt.plot([100,50,200,150],vmeanL1normGRAYMask,'g+',label = 'GrayMask_Disc1.5')
plt.ylabel('L1norm_mask')
plt.xlabel('epochs')
plt.axis([0,210,5,7])
plt.legend()

plt.subplot(224)
L2norm_mask = plt.plot([100,50,200,150],vmeanL2normGRAYMask,'g+',label = 'GrayMask_Disc1.5')
plt.ylabel('L1norm_mask')
plt.xlabel('epochs')
plt.axis([0,210,0.2,0.4])
plt.legend()

plt.show()





