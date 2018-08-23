from pdb import set_trace as st
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image setup')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str)
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str)
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str)

for arg in vars(args):
	print('[%s] = ' % arg,  getattr(args, arg))

splits = filter( lambda f: not f.startswith('.'), os.listdir(args.fold_A)) # ignore hidden folders like .DS_Store

for sp in splits:
	img_fold_A = os.path.join(args.fold_A, sp)
	img_fold_B = os.path.join(args.fold_B, sp)
	img_list = filter( lambda f: not f.startswith('.'), os.listdir(img_fold_A)) # ignore hidden folders like .DS_Store
	num_imgs = len(img_list)
	print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
	img_fold_AB = os.path.join(args.fold_AB, sp)
	if not os.path.isdir(img_fold_AB):
		os.makedirs(img_fold_AB)
	print('split = %s, number of images = %d' % (sp, num_imgs))
	for n in range(num_imgs):
		name_A = img_list[n]
		path_A = os.path.join(img_fold_A, name_A)
		name_B = name_A
		path_B = os.path.join(img_fold_B, name_B)
		if os.path.isfile(path_A) and os.path.isfile(path_B):
			name_AB = name_A
			path_AB = os.path.join(img_fold_AB, name_AB)
			im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
			im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
			im_AB = np.concatenate([im_A, im_B], 1)
			cv2.imwrite(path_AB, im_AB)
