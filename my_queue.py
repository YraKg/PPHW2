#
#   @date:  [2/3/2024]
#   @author: [harel krelbaum and yurii kohan]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#

from multiprocessing import Process, Lock, Pipe
import numpy as np
from typing import List, Tuple
class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.q = []
        self.len = 0
        self.lock = Lock()
       # raise NotImplementedError("To be implemented")

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.q.insert(0, msg)
        self.len += 1
        p = Pipe()
        p.

      #  raise NotImplementedError("To be implemented")

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        self.lock.acquire()
        Pipe
        try:
           msg = self.q.pop()
           self.len -= 1
        finally:
            self.lock.release()
            return msg
        #raise NotImplementedError("To be implemented")
    
    def length(self):
        '''Get the number of messages currently in the queue
            
        Return
        ------
        An integer
        '''
        return self.len
        #raise NotImplementedError("To be implemented")

    def empty(self):
        return self.len == 0

