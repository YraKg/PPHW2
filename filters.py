#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from numba import cuda
from numba import njit
import imageio
import matplotlib.pyplot as plt
import numpy as np

def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    raise NotImplementedError("To be implemented")


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations

    kernel_0 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_1 = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
    kernel_2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_4 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    sobel_filter = kernel_0

    G_x = correlation_numba(sobel_filter, pic)
    G_y = correlation_numba(np.transpose(sobel_filter), pic)

    G_x_2 = np.square(G_x)
    G_y_2 = np.square(G_y)
    G_sum = G_x_2 + G_y_2
    G_sqrt = np.sqrt(G_sum)

    return G_sqrt




def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
