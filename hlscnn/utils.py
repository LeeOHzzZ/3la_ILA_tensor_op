"""This file contains utils used in the code generation for HLSCNN"""

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
        self.__counter = [0] * N
        self.__max_value = tuple(loopbounds)

    @property
    def value(self):
        return self.__counter
    
    @property
    def maxValue(self):
        return self.__max_value
    
    @value.setter
    def value(self, value):
        self.__counter = value

    def reset(self):
        self.__counter = tuple(0 for i in range(self.N))
    
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
        return self.__increment_ith(self.N-1)
        