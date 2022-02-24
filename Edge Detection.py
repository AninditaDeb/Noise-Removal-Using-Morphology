import cv2
from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here
     
    m,n= img.shape
    constant=1
    denoise_img= np.zeros((m,n), dtype=np.uint8)
    for i in range(constant, m):
        for j in range(constant,n):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            denoise_img[i,j]= np.median(temp)
    #raise NotImplementedError
    return denoise_img
def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """   
    denoise_img=filter(img)
    edge_x_normalized, edge_y_normalized, edge_mag=edge_detect(denoise_img)
    edge_45_normalized,edge_135_normalized= edge_diag(denoise_img)
    return edge_x_normalized,edge_y_normalized,edge_mag,edge_45_normalized,edge_135_normalized
    # TO DO: implement your solution here
   
    
   # raise NotImplementedError
    return conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """
    # TO DO: implement your solution here
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)
    sobel_x_flipped=np.flip(sobel_x)
    sobel_y_flipped=np.flip(sobel_y)
    image_h=img.shape[0]
    image_w=img.shape[1]
    kernel_h=sobel_x_flipped.shape[0]
    kernel_w=sobel_x_flipped.shape[1]
    h=kernel_h//2
    w=kernel_w//2
    edge_x=np.zeros(img.shape)
    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum=0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum=(sum+sobel_x_flipped[m][n]*img[i-h+m][j-w-n])
                edge_x[i][j]=sum
    
    kernel_p=sobel_y_flipped.shape[0]
    kernel_q=sobel_y_flipped.shape[1] 
    p=kernel_p//2
    q=kernel_q//2
    edge_y=np.zeros(img.shape)
    for i in range(p,image_h-p):
        for j in range(q,image_w-q):
            sum=0
            for m in range(kernel_p):
                for n in range(kernel_q):
                    sum=(sum+sobel_y_flipped[m][n]*img[i-p+m][j-q-n])
                edge_y[i][j]=sum
    edge_mag=np.zeros(edge_x.shape)
    for i in range(edge_x.shape[0]):
        for j in range(edge_x.shape[1]):
            q=(edge_x[i][j]**2+edge_y[i][j]**2)**(1/2)
            edge_mag[i][j]=q

    #raise NotImplementedError
    edge_x_normalized=255*((edge_x-np.min(edge_x))/(np.max(edge_x)-np.min(edge_x)))
    edge_y_normalized=255*((edge_y-np.min(edge_y))/(np.max(edge_y)-np.min(edge_y)))
    return edge_x_normalized, edge_y_normalized, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """
    sobel_45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(int)
    sobel_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).astype(int)
    # TO DO: implement your solution here
    sobel_x_diagonal_45_flipped=np.flip(sobel_45)
    sobel_x_diagonal_135_flipped=np.flip(sobel_135)
    image_h=img.shape[0]
    image_w=img.shape[1]
    kernel_h=sobel_x_diagonal_45_flipped.shape[0]
    kernel_w=sobel_x_diagonal_45_flipped.shape[1]
    h=kernel_h//2
    w=kernel_w//2
    edge_45=np.zeros(img.shape)
    for i in range(h,image_h-h):
        for j in range(w,image_w-w):
            sum=0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum=(sum+sobel_x_diagonal_45_flipped[m][n]*img[i-h+m][j-w-n])
                edge_45[i][j]=sum
    
    kernel_p=sobel_x_diagonal_135_flipped.shape[0]
    kernel_q=sobel_x_diagonal_135_flipped.shape[1] 
    p=kernel_p//2
    q=kernel_q//2
    edge_135=np.zeros(img.shape)
    for i in range(p,image_h-p):
        for j in range(q,image_w-q):
            sum=0
            for m in range(kernel_p):
                for n in range(kernel_q):
                    sum=(sum+sobel_x_diagonal_135_flipped[m][n]*img[i-p+m][j-q-n])
                edge_135[i][j]=sum

    #raise NotImplementedError
    edge_45_normalized=255*((edge_45-np.min(edge_45))/(np.max(edge_45)-np.min(edge_45)))
    edge_135_normalized=255*((edge_135-np.min(edge_135))/(np.max(edge_135)-np.min(edge_135)))
   

    #raise NotImplementedError
    print("The kernel along forty five degrees direction from x axis of the image is {}".format(sobel_45))
    print("The kernel along one thirty five degrees direction from x axis of the image is {}".format(sobel_135))# print the two kernels you designed here
    return edge_45_normalized,edge_135_normalized


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    cv2.imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    cv2.imwrite('results/task2_edge_x.jpg', edge_x_img)
    cv2.imwrite('results/task2_edge_y.jpg', edge_y_img)
    cv2.imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    cv2.imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    cv2.imwrite('results/task2_edge_diag2.jpg', edge_135_img)





