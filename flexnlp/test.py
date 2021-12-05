import sys
import os
import subprocess

from linear_layer_driver import linear_layer_driver
from lstm_driver import lstm_layer_driver
from pooling_driver import pooling_layer_driver
from layernorm_driver import layernorm_driver
from attention_driver import attention_layer
from src.utils import tool

def test_lstm():
  assert len(sys.argv) >= 7, \
    "Usage: python3 test.py lstm [num_vector_in] [num_vector_out] [num_timestep] [is_bias] [is_zero_first]"
  num_v_in = int(sys.argv[2])
  num_v_out = int(sys.argv[3])
  num_ts = int(sys.argv[4])
  is_bias = int(sys.argv[5])
  is_zero_first = int(sys.argv[6])
  if len(sys.argv) > 7:
    iter_num = int(sys.argv[7])
  else:
    iter_num = 1
  test_driver = lstm_layer_driver(num_v_in, num_v_out, num_ts, is_bias, is_zero_first)
  use_relay = True
  verbose_analysis = 0
  err_out_list = []
  for i in range(iter_num):
    err_out_list.append(test_driver.run_test(use_relay, verbose_analysis)[0])
  mean, stdd = tool().cal_mean_stdd(err_out_list)
  print(f"Summary of Mismatch: Mean: {mean:.5%}\t Standard Deviation: {stdd:.5%}.")
  test_driver.clean_up()

def test_linear_layer():
  assert len(sys.argv) >= 6, \
    "Usage: python3 test.py linear_layer [num_vector_in] [num_vector_out] [num_timestep] [is_bias] [dtype]"
  num_v_in = int(sys.argv[2])
  num_v_out = int(sys.argv[3])
  num_ts = int(sys.argv[4])
  is_bias = int(sys.argv[5])
  if len(sys.argv) > 6:
    dtype = str(sys.argv[6])
  else:
    print("Using default dtype: float32")
    dtype = "float32"

  test_driver = linear_layer_driver(num_v_in, num_v_out, num_ts, is_bias, dtype, "linear_layer_test")
  test_driver.run_test()
  test_driver.clean_up()

def test_pooling():
  assert len(sys.argv) == 5, \
    "Usage: python3 test.py pooling [mode] [num_vector_in] [num_timestep]"
  mode = sys.argv[2]
  num_v_in = int(sys.argv[3])
  num_ts = int(sys.argv[4])

  test_driver = pooling_layer_driver(mode, num_v_in, num_ts)
  verbose_analysis = 0
  test_driver.run_test(verbose_analysis)
  test_driver.clean_up()

def test_layernorm():
  assert len(sys.argv) == 4, \
    "Usage: python3 test.py layernorm [num_vector_in] [num_timestep]"
  num_v = int(sys.argv[2])
  num_ts = int(sys.argv[3])
  test_driver = layernorm_driver(num_v, num_ts)
  verbose_analysis = 0
  test_driver.run_test(verbose_analysis)
  test_driver.clean_up()

def test_attention_layer():
  assert len(sys.argv) >= 4, \
    "Usage: python3 test.py attention [num_ts] [num_v]"
  num_ts = int(sys.argv[2])
  num_v = int(sys.argv[3])
  if len(sys.argv) > 4:
    loop_num = int(sys.argv[4])
  else:
    loop_num = 1
  test_driver = attention_layer(num_ts=num_ts, num_v=num_v, mem_idx_enc=0, mem_idx_dec=0)
  err_out_list = []
  for i in range(loop_num):
    err_out_list.append(test_driver.run_test())
  mean, stdd = tool().cal_mean_stdd(err_out_list)
  print(f"Summary of Mismatch: Mean: {mean:.5%}\t Standard Deviation: {stdd:.5%}.")

if __name__ == '__main__':
  test_name = sys.argv[1]
  supported_test = ('lstm', 'linear_layer', 'pooling', 'layernorm', 'attention', 'all')
  assert test_name in supported_test, \
    '{} is not supported, supported test is {}'.format(test_name, supported_test)
  
  if test_name == 'lstm':
    test_lstm()
  if test_name == 'linear_layer':
    test_linear_layer()
  if test_name == 'pooling':
    test_pooling()
  if test_name == 'layernorm':
    test_layernorm()
  if test_name == 'attention':
    test_attention_layer()
  if test_name == 'all':
    print("run all the test with default parameters")
    err_out_list = []
    for i in range(10):
      err_out_list += lstm_layer_driver(num_v_in = 16, num_v_out = 16, num_ts = 1, 
                      is_bias = 1, is_zero_first = 1).run_test(use_relay=0)
    err_out_list += linear_layer_driver(num_v_in = 16, num_v_out = 16, num_ts = 10, is_bias = 1).run_test()
    err_out_list += pooling_layer_driver('max', num_v_in=16, num_ts=10).run_test()
    err_out_list += pooling_layer_driver('mean', num_v_in=16, num_ts=10).run_test()
    err_out_list += layernorm_driver(num_v=16, num_ts=10).run_test()
    err_out_list += attention_layer(num_v=16, num_ts=10, mem_idx_enc=0, mem_idx_dec=0)
    print("average result mismatch with ref is {:04%}".format(sum(err_out_list)/len(err_out_list)))
    # clean up
    for file in os.listdir('./test'):
      if '.tmp' in file:
        subprocess.run(['rm', './test/'+file])
