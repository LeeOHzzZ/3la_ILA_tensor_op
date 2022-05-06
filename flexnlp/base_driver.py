""" This file contains a base driver for all the operator codegens
"""

class FlexASRBaseDriver():
    
    def __init__(self):
        self.ADPTFLOAT_N_BITS = 8
        self.ADPTFLOAT_N_EXP = 3
        self.ADPTFLOAT_OFFSET=10