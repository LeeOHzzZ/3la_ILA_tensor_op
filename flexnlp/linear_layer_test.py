import json
import sys
import numpy as np
import subprocess
import os

from utils import tool as tool
from converter import Converter as cvtr
from linear_layer_driver import linear_layer_driver as driver

if __name__ == '__main__':
  assert len(sys.argv) == 5, \
    "Usage: python3 linear_layer_testflow.py [num_vector_in] [num_vector_out] [num_timestep] [is_bias]"
  num_v_in = int(sys.argv[1])
  num_v_out = int(sys.argv[2])
  num_ts = int(sys.argv[3])
  is_bias = int(sys.argv[4])

  test = driver(num_v_in, num_v_out, num_ts, is_bias)
  test.run_test()
  
  test.wgt.tofile('./data/wgt.txt', sep = '\n')
  test.inp.tofile('./data/inp.txt', sep = '\n')
  test.bias.tofile('./data/bias.txt', sep = '\n')

  test.clean_up()
