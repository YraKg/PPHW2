#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from numba import njit, cuda, prange
import imageio
import matplotlib.pyplot as plt
import numpy as np
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
    kernel_global_mem = cuda.to_device(kernel)

    image_global_mem = cuda.to_device(image)

    result_global_mem = cuda.device_array((image.shape[0], image.shape[1]))

    threadsperblock =  (28,28)

    blockspergrid = (1,1)

    correlation_kernel[blockspergrid, threadsperblock](kernel_global_mem, image_global_mem, result_global_mem)

    result = result_global_mem.copy_to_host()
    return result


@cuda.jit
def correlation_kernel(kernel, image, result):

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    image_row = tx - ((kernel.shape[0] - 1) / 2)
    image_column = ty - ((kernel.shape[1] - 1) / 2)

    sum = 0

    for kernel_row in range(kernel.shape[0]):
        curr_image_row = image_row + kernel_row
        if (curr_image_row < 0 or curr_image_row >= image.shape[0]):
            continue
        for kernel_column in range(kernel.shape[1]):
            curr_image_column = image_column + kernel_column
            if (curr_image_column < 0 or curr_image_column >= image.shape[1]):
                continue
            sum += kernel[kernel_row][kernel_column] * image[curr_image_row][curr_image_column]

    result[tx][ty] = sum

@njit(parallel = True)
def calc_correlation(kernel, image, image_row , image_column):  #gets the top left of a matrix as a parameter
    sum = 0
    curr_image_row = 0
    curr_image_column = 0

    for kernel_row in prange(kernel.shape[0]) :
        curr_image_row = image_row + kernel_row
        if(curr_image_row < 0 or curr_image_row >= image.shape[0]):
            continue
        for kernel_column in prange(kernel.shape[1]):
            curr_image_column = image_column + kernel_column
            if(curr_image_column < 0 or curr_image_column >= image.shape[1]):
                continue
            sum += kernel[kernel_row][kernel_column] * image[curr_image_row][curr_image_column]

    return sum

@njit(parallel = True)
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
    result = np.zeros((image.shape[0], image.shape[1]))
    top_left_row = 0
    top_left_column = 0
    for i in prange(image.shape[0]):
        for j in prange(image.shape[1]):
            #they wrote in the piazza that we are allowed to assume that the kernel matrix has an odd number of rows and columns
            top_left_row = i - ((kernel.shape[0] - 1) / 2)
            top_left_column = j - ((kernel.shape[1] - 1) / 2)
            result[i][j] = calc_correlation(kernel, image, top_left_row, top_left_column)
    return result


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
