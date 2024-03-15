#
#   @date:  [2/3/2024]
#   @author: [harel krelbaum and yurii kohan]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing
from multiprocessing import Process, Lock, Pipe, Queue
import numpy as np
from typing import List, Tuple
class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.p_w, self.p_r = Pipe()

        self.writer_lock = Lock()


    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''

        print("i am putting an image here")
        self.writer_lock.acquire(block=True)

        self.p_w.send(msg)

        self.writer_lock.release()

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''

        print("i am getting an image")
        while self.empty():
            continue

        self.writer_lock.acquire(block=True)

        msg = self.p_r.recv()

        self.writer_lock.release()

        return msg

    # def length(self):
    #     '''Get the number of messages currently in the queue
    #
    #     Return
    #     ------
    #     An integer
    #     '''
    #
    #     return self.len

    def empty(self):
        print("am i empty?"+str(not self.p_w.poll()))
        return not self.p_w.poll()

