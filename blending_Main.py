##############################################################################
"""
BLENDING

Abdulmajeed Muhammad Kabir
2018
"""

import cv2
import blending as bl

path1 = r"E:\Desky\orange.jpg"
path2 = r"E:\Desky\apple.jpg"
path3 = r"E:\Desky\mask.jpg"


image1 = cv2.imread(path1, cv2.IMREAD_COLOR)
image2 = cv2.imread(path2, cv2.IMREAD_COLOR)
mask   = cv2.imread(path3, cv2.IMREAD_COLOR)

align  = bl.alignment([image1, image2, mask])
image1 = align[0] 
image2 = align[1]
mask   = align[2]

mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

finalImage = bl.measures_fusion_multires(image1, image2, mask, levels=1191)
cv2.imshow('Image1', image1)
cv2.imshow('Image2', image2)
cv2.imshow('Mask', mask)
cv2.imshow('Blended', finalImage)
cv2.waitKey(1);
cv2.destroyAllWindows()


