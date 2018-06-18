import cv2
import numpy as np

im1 = cv2.imread('/home/berta/Documents/CoRL/input_4.png',0)
mask1 = cv2.imread('/home/berta/Documents/CoRL/mask_4.png',0)
im2 = cv2.imread('/home/berta/Documents/CoRL/input_5.png',0)
mask2 = cv2.imread('/home/berta/Documents/CoRL/mask_5.png',0)

kernel = np.ones((15,15),np.uint8)

mask1 = cv2.resize(mask1, (256,256))
mask1 = cv2.dilate(mask1,kernel,iterations = 1)
mask2 = cv2.resize(mask2, (256,256))
mask2 = cv2.dilate(mask2,kernel,iterations = 1)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
keypoints1, descriptor1 = orb.detectAndCompute(im1,None)
keypoints2, descriptor2 = orb.detectAndCompute(im2,None)

keypoints1_mask = list()
descriptor1_mask = np.empty(shape=descriptor1.shape,dtype=np.uint8)
j = 0

for i in range(0,len(keypoints1)):
	if (mask1[int(keypoints1[i].pt[0]),int(keypoints1[i].pt[1])] == 0):
		keypoints1_mask.append(keypoints1[i])
		descriptor1_mask[j] = descriptor1[i]
		j = j + 1

descriptor1_mask = descriptor1_mask[:j]

keypoints2_mask = list()
descriptor2_mask = np.empty(shape=descriptor2.shape,dtype=np.uint8)
j = 0

for i in range(0,len(keypoints2)):
	if (mask2[int(keypoints2[i].pt[0]),int(keypoints2[i].pt[1])] == 0):
		keypoints2_mask.append(keypoints2[i])
		descriptor2_mask[j] = descriptor2[i]
		j = j +	1

descriptor2_mask = descriptor2_mask[:j]

keypoints2 = keypoints2_mask
keypoints1 = keypoints1_mask
descriptor1 = descriptor1_mask
descriptor2 = descriptor2_mask

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(descriptor1,descriptor2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
 
# Draw first 10 matches.
im = np.zeros((1,1))
im12 = cv2.drawMatches(im1,keypoints1,im2,keypoints2,matches[:40],im,flags=2)

cv2.imwrite('/home/berta/Documents/CoRL/matchesTargetMask4TargetMask5.png', im12)