"""This file contains utils used in the code generation for HLSCNN"""
from functools import reduce
from math import floor
from operator import mul

class LoopCounter():
    """This class models a loop counter of iteration of the N nested loops"""
    def __init__(self, N, loopbounds):
        """Required inputs:
            - N: Total number of the nested loops
            - loopbounds: The loop bounds of the nested N loops
        """
        assert len(loopbounds) == N, "N is not equal to the length of the loop bounds"
        assert all(i > 0 for i in loopbounds), "loop bounds contains non-positive numbers"
        self.N = N
        self.__cntr = 0
        self.__counter = [0] * N
        self.__max_value = tuple(loopbounds)
        self.__strides = tuple([reduce(mul, loopbounds[i+1:]) for i in range(N-1)] + [1])
        print(self.__strides)

    @property
    def value(self):
        return self.__counter
    
    @property
    def maxValue(self):
        return self.__max_value
    
    @property
    def cntr(self):
        return self.__cntr
    
    @value.setter
    def value(self, value):
        self.__counter = value
        self.__cntr = sum(map(mul, self.__counter, self.__strides))

    def reset(self):
        self.__counter = tuple(0 for i in range(self.N))
        self.__cntr = 0
    
    def __increment_ith(self, i):
        """recursively increment the counter values
            It returns True when correctly increment the counter values,
            else it returns False if overflows
        """
        assert i < self.N
        if (i==0) and (self.__counter[i] == self.__max_value[i] - 1):
            self.__counter[i] = 0
            return False
        elif (i > 0) and (self.__counter[i] == self.__max_value[i] - 1):
            self.__counter[i] = 0
            return self.__increment_ith(i-1)
        else:
            self.__counter[i] += 1
            return True 
    
    def increment(self):
        self.__cntr += 1
        return self.__increment_ith(self.N-1)
        