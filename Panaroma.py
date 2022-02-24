import cv2 
from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np
import random
import matplotlib.pyplot as plt
def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    finalH, inliers = ransac(left_img,right_img,5)
    result_img = blending(left_img,right_img,finalH)
    return result_img
def calculate_right_left_descriptors(left_img,right_img):
    sift = cv2.SIFT_create()
    lp,left_des = sift.detectAndCompute(left_img,None)
    rp,right_des= sift.detectAndCompute(right_img,None)
    return lp,left_des,rp,right_des
    
    #######Function would compute KNN between left anf right images and return 2 best matches##
def compute_KNN_lefttoright(left_des,right_des):
    distance_list=[]
    distance_list1=[]
    k=0
    for row in left_des:
        distance=np.sqrt(np.sum(np.square(row-right_des),axis=1))
        t=0
        for i in range(len(distance)):
            distance_list.append((row,right_des[i],distance[i],k,t))
            t=t+1
        distance_list.sort(key=lambda tup: tup[2])
        for i in range(2):
            distance_list1.append(distance_list[i])
        distance_list=[]
        k=k+1
    distance_list1.sort(key=lambda tup: tup[2])
    return distance_list1

#######Function would compute KNN between right and left images and return 2 best matches##
def compute_KNN_righttoleft(left_des,right_des):
    distance_list=[]
    distance_list2=[]
    k=0
    for row in right_des:
        distance=np.sqrt(np.sum(np.square(row-left_des),axis=1))
        t=0
        for i in range(len(distance)):
            distance_list.append((row,left_des[i],distance[i],k,t))
            t=t+1
        distance_list.sort(key=lambda tup: tup[2])
        for i in range(2):
            distance_list2.append(distance_list[i])
        distance_list=[]
        k=k+1
    distance_list2.sort(key=lambda tup: tup[2])
    return distance_list2
############Function calculates whether pairs obtained from compute_KNN_lefttoright and compute_KNN_righttoleft are matching 
##############or not and if not matching then removes those ambiguous pairs
def cross_check(distance_list1,distance_list2):
    cross_check_pairs=[]
    if len(distance_list1)<len(distance_list2):
        for i in range(len(distance_list1)):
            if (distance_list1[i][2]==distance_list2[i][2])and (distance_list1[i][3]==distance_list2[i][4])and(distance_list1[i][4]==distance_list2[i][3]):
                cross_check_pairs.append(distance_list1[i])
            #print (distance_list1[i])
    else:
        for i in range(len(distance_list2)):
            if (distance_list1[i][2]==distance_list2[i][2])and (distance_list1[i][3]==distance_list2[i][4])and(distance_list1[i][4]==distance_list2[i][3]):
                cross_check_pairs.append(distance_list1[i])
    return cross_check_pairs

def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#

def ransac(left_img,right_img,thresh):
    
    lp,left_des,rp,right_des=calculate_right_left_descriptors(left_img,right_img)
    keypoints=[rp,lp]
    distance_list1=compute_KNN_lefttoright(left_des,right_des)
    distance_list2=compute_KNN_righttoleft(left_des,right_des)
    cross_check_pairs=cross_check(distance_list1,distance_list2)
    correspondenceList=[]
    for i in range(len(cross_check_pairs)):
        (x1, y1) = keypoints[0][cross_check_pairs[i][4]].pt
        (x2, y2) = keypoints[1][cross_check_pairs[i][3]].pt
        correspondenceList.append([x1,y1,x2,y2])
    corr=np.matrix(correspondenceList)
    maxInliers = []
    finalH = None
    for i in range(5000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        #print "Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers)

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers
################Blending the two images ####################################
def create_mask(left_img,right_img,version):
    height_img1 = left_img.shape[0]
    width_img1 = left_img.shape[1]
    width_img2 = right_img.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2
    offset = int(800 / 2)
    barrier = left_img.shape[1] - int(800 / 2)
    mask = np.zeros((height_panorama, width_panorama))
    if version== 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])

def blending(img1,img2,H):
    H = H
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 =create_mask(img1,img2,version='left_image')
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    return final_result
if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img =cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
