import cv2
import numpy as np
import os

input_path = '/home/bescosb/pix2pix_0.1/results/CARLA/RGB_Dyn_Stat/150_net_G_test/images/input/'
mask_path = '/home/bescosb/CARLA_0.8.2/dataset/MaskDyn/test/'
output_geometry_path = '/home/bescosb/pix2pix_0.1/results/CARLA/geometry/'

for filename in os.listdir(input_path):
	image = cv2.imread(os.path.join(input_path,filename),0)
	mask = cv2.imread(os.path.join(mask_path,filename),0)

	mask = cv2.resize(mask, (256,256))
	output_geometry = cv2.inpaint(image, mask,3,cv2.INPAINT_TELEA)

	cv2.imwrite(os.path.join(output_geometry_path,filename),output_geometry)
