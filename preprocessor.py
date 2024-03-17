#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing
import numpy as np
import scipy as sp

imageDim = (28,28)
vectorImageDim = (784)
maxPixel = 255
minPixel = 0

class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, num_augmentations):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''

        self.jobs = jobs
        self.result = result

        self.num_augmentations = num_augmentations

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''

        reshaped = np.reshape(image, imageDim)

        rotated = sp.rotate(reshaped, angle, reshape=False)

        return np.reshape(rotated, vectorImageDim)



    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''

        reshaped = np.reshape(image,imageDim)

        shifted = sp.ndimage.shift(reshaped, [dx, dy])

        return np.reshape(shifted, vectorImageDim)
    
    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''

        added_noise = image + np.random.uniform(low=-noise, high=noise, size=vectorImageDim)

        # take care of values that exceed [0,255] bounds
        upper_bound = np.minimum(added_noise, 255)
        lower_bound = np.maximum(upper_bound,0)

        return lower_bound

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''

        result = np.reshape(image,imageDim)

        rows, cols = imageDim

        #if tilt is negative should iterate rows in inverted direction to use vals before update
        neg_tilt = -1 if tilt < 0 else 1

        for i in range(rows):
            for j in range(0,cols,neg_tilt):

                shifted_j = int(j+i*tilt)

                if shifted_j >= cols or shifted_j < 0:
                    result[i][j] = 0.
                else:
                    result[i][j] = result[i][shifted_j]

        return np.reshape(result, vectorImageDim)

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''

        # define range for preprocessing parameters
        angle_range = [-15, 15]
        shift_range_x = [-1, 2]
        shift_range_y = [-1, 2]
        noise_range = [1, 1]
        tilt_range = [0.0, 0.5]

        # randomize parameters for preprocessing
        angle = int(np.random.randint(low=angle_range[0], high=angle_range[1]))
        dx = int(np.random.randint(low=shift_range_x[0], high=shift_range_x[1]))
        dy = int(np.random.randint(low=shift_range_y[0], high=shift_range_y[1]))
        noise = int(np.random.randint(low=noise_range[0], high=noise_range[1]))
        tilt = int(np.random.uniform(low=tilt_range[0], high=tilt_range[1]))

        # define order of preprocessing functions
        order = [4, 2, 1, 3]

        # run image through preprocessing functions in given order
        for i in order:
            if i == 1:
                image = Worker.rotate(image,angle)
            elif i == 2:
                image = Worker.shift(image,dx,dy)
            elif i == 3:
                image = Worker.add_noise(image, noise)
            elif i == 4:
                image = Worker.skew(image,tilt)

        return image


    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        num_augmentations = self.num_augmentations - 1

        image, label = self.jobs.get()
        while image is not None:
            for i in range(num_augmentations):
                new_image = self.process_image(image)
                new_labeled = (new_image, label)
                self.result.put(new_labeled)
            self.result.put((image,label))
            self.jobs.task_done()

            image, label = self.jobs.get()


        self.jobs.task_done()
