#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import math
import multiprocessing
import os
import preprocessor as pp
import my_queue
from network import *

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''

        num_augmentations = 1
        #TODO: change this

        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        data_queue = multiprocessing.JoinableQueue()
        result_queue = my_queue.MyQueue()

        num_workers = int(os.getenv('SLURM_CPUS_PER_TASK',1))
        workers = [pp.Worker(data_queue,result_queue,num_augmentations) for i in range(num_workers)]
        for w in workers:
            w.start()

        # 2. Set jobs

        for i in zip(training_data[0],training_data[1]):
            data_queue.put(i)
        for i in range(num_workers):
            data_queue.put((None,None))

        data_queue.join()

        data = ([],[])
        num_datapoints = len(training_data[0]) * num_augmentations

        while num_datapoints:
            point = result_queue.get()
            data[0].append(point[0])
            data[1].append(point[1])
            num_datapoints-=1


        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(data, validation_data)

        # 3. Stop Workers






    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''

        size = math.ceil(len(data[1])/self.number_of_batches)

        return super().create_batches(data, labels,size)
