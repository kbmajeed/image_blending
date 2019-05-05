"""
BLENDING
"""


"""
Dependencies
"""
###############################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
###############################################################################





"""
Check and Align Images by Size
"""   
###############################################################################
def alignment(image_stack):
#'-----------------------------------------------------------------------------#
    sizes = []
    D = len(image_stack)
    for i in range(D):
        sizes.append(np.shape(image_stack[i]))
    sizes = np.array(sizes)
    for i in range(D):
        if np.shape(image_stack[i])[:2] !=  (min(sizes[:,0]),min(sizes[:,1])):
            print("Detected Non-Uniform Sized Image"+str(i)+" ... Resolving ...")
            image_stack[i] = cv2.resize(image_stack[i], (min(sizes[:,1]), min(sizes[:,0])))
            print(" *Done")
    print("\n")
    return image_stack
###############################################################################





"""
Laplacian and Gaussian Pyramids
""" 
###############################################################################
def multires_pyramid(image, levels):
#'-----------------------------------------------------------------------------#
    levels  = levels - 1
    imgGpyr = [image]
    for i in range(levels):
        imgW = np.shape(imgGpyr[i])[1]
        imgH = np.shape(imgGpyr[i])[0]
        imgGpyr.append(cv2.pyrDown(imgGpyr[i].astype('float64')))
    imgLpyr = [imgGpyr[levels]]
    for i in range(levels, 0, -1):
        imgW = np.shape(imgGpyr[i-1])[1]
        imgH = np.shape(imgGpyr[i-1])[0]
        imgLpyr.append(imgGpyr[i-1] - cv2.resize(cv2.pyrUp(imgGpyr[i]),(imgW,imgH)))
    return imgLpyr[::-1], imgGpyr
###############################################################################





"""
Multiresolution Measures Fusion
""" 
###############################################################################
def measures_fusion_multires(image1, image2, mask, levels=6):
#'-----------------------------------------------------------------------------#
    imgLpyr1, imgGpyr1 = multires_pyramid(image1, levels) 
    imgLpyr2, imgGpyr2 = multires_pyramid(image2, levels)   
    wLpyr, wGpyr    = multires_pyramid(mask, levels)   
    blendedPyramids = []
    for j in range(levels):
        ii1 = imgLpyr1[j].astype('float64')
        ii2 = imgLpyr2[j].astype('float64')
        www = wGpyr[j]#.astype('float64')
        blendedPyramids.append( www*ii1 + (1-www)*ii2 )
        #blendedPyramids.append( (1-www)*ii1 + (www)*ii2 )
    finalImage = []
    blended_final = np.array(blendedPyramids[0])
    for i in range(levels-1):
        imgH = np.shape(image1)[0]; 
        imgW = np.shape(image2)[1]; 
        layerx = cv2.pyrUp(blendedPyramids[i+1])
        blended_final += cv2.resize(layerx,(imgW,imgH))
    blended_final[blended_final < 0] = 0
    blended_final[blended_final > 255] = 255
    
    finalImage.append(blended_final)
    output = finalImage[0]
    #output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(" *Done"); print("\n")
    return output.astype('uint8')
###############################################################################





