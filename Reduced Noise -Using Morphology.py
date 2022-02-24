from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import cv2
import numpy as np
def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    structuring_element=np.array([[1,1,1],[1,1,1],[1,1,1]])
    padded=len(img)%len(structuring_element)
    x=np.zeros((padded,len(img)))
    img=np.append(img,x,axis=0)
    y=np.zeros((len(img),padded))
    img=np.append(img,y,axis=1)
    img.shape
    m,n= img.shape
    structuring_element=np.array([[1,1,1],[1,1,1],[1,1,1]])
    constant= (len(structuring_element)-1)//2
    erode_img= np.zeros((m,n), dtype=np.uint8)
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*structuring_element
            erode_img[i,j]= np.min(product)
    #raise NotImplementedError
    return erode_img
def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    #Define new image to store the pixels of dilated image
    p,q= img.shape
    structuring_element=np.array([[1,1,1],[1,1,1],[1,1,1]])
    dilate_img= np.zeros((p,q), dtype=np.uint8)

    #Define the structuring element 
    constant=1
    #Dilation operation without using inbuilt CV2 function
    for i in range(constant, p-constant):
        for j in range(constant,q-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*structuring_element
            dilate_img[i,j]= np.max(product)
    
    #raise NotImplementedError
    return dilate_img
def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    ####Opening operation#########
    eroded_image = morph_erode(img)
    open_img  =  morph_dilate(eroded_image)
   # raise NotImplementedError
    return open_img


def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    # TO DO: implement your solution here
    dialate_img=morph_dilate(img)
    close_img=morph_erode(dialate_img)
    #raise NotImplementedError
    return close_img

def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    open_image=morph_open(img)
    denoise_img=morph_close(open_image)
    #raise NotImplementedError
    return denoise_img

def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    eroded_image = morph_erode(img)
    structuring_element=np.array([[1,1,1],[1,1,1],[1,1,1]])
    padded=len(img)%len(structuring_element)
    x=np.zeros((padded,len(img)))
    img=np.append(img,x,axis=0)
    y=np.zeros((len(img),padded))
    img=np.append(img,y,axis=1)
    bound_img=img-eroded_image
    #raise NotImplementedError
    return bound_img
if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)
