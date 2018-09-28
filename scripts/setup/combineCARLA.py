# Copyright (C) 2018 Berta Bescos 
# <bbescos at unizar dot es> (University of Zaragoza)
#

from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse
import skimage.io

parser = argparse.ArgumentParser('create image setup')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str)
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str)
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C', type=str)
parser.add_argument('--fold_ABC', dest='fold_ABC', help='output directory', type=str)

for arg in vars(args):
	print('[%s] = ' % arg,  getattr(args, arg))

splits = filter( lambda f: not f.startswith('.'), os.listdir(args.fold_A)) # ignore hidden folders like .DS_Store

for sp in splits:
	img_fold_A = os.path.join(args.fold_A, sp)
	img_fold_B = os.path.join(args.fold_B, sp)
	img_fold_C = os.path.join(args.fold_C, sp)
	img_list = filter( lambda f: not f.startswith('.'), os.listdir(img_fold_A)) # ignore hidden folders like .DS_Store
	num_imgs = len(img_list)
	print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
	img_fold_ABC = os.path.join(args.fold_ABC, sp)
	if not os.path.isdir(img_fold_ABC):
		os.makedirs(img_fold_ABC)
	print('split = %s, number of images = %d' % (sp, num_imgs))
	for n in range(num_imgs):
		name_A = img_list[n]
		path_A = os.path.join(img_fold_A, name_A)
		name_B = name_A
		name_C = name_A
		path_B = os.path.join(img_fold_B, name_B)
		path_C = os.path.join(img_fold_C, name_C)
		if os.path.isfile(path_A) and os.path.isfile(path_B) and os.path.isfile(path_C):
			name_ABC = name_A
			path_ABC = os.path.join(img_fold_ABC, name_ABC)
			im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
			im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
			im_C = skimage.io.imread(path_C)
			_im_C = im_C.copy()
			_im_C[:,:,:] = 0
			_im_C[im_C[:,:,0] == 4] = 255		
			_im_C[im_C[:,:,0] == 10] = 255
			im_C = _im_C.copy()
			im_ABC = np.concatenate([im_A, im_B, im_C], 1)
			cv2.imwrite(path_ABC, im_ABC)
