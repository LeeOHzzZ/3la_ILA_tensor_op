import sys

from linear_layer_driver import linear_layer_driver
from lstm_driver import lstm_layer_driver
from pooling_driver import pooling_layer_driver

def test_lstm():
  assert len(sys.argv) == 7, \
    "Usage: python3 test.py lstm [num_vector_in] [num_vector_out] [num_timestep] [is_bias] [is_zero_first]"
  num_v_in = int(sys.argv[2])
  num_v_out = int(sys.argv[3])
  num_ts = int(sys.argv[4])
  is_bias = int(sys.argv[5])
  is_zero_first = int(sys.argv[6])
  test_driver = lstm_layer_driver(num_v_in, num_v_out, num_ts, is_bias, is_zero_first)
  use_relay = 1
  verbose_analysis = 0
  test_driver.run_test(use_relay, verbose_analysis)
  test_driver.clean_up()

def test_linear_layer():
  assert len(sys.argv) == 6, \
    "Usage: python3 test.py linear_layer [num_vector_in] [num_vector_out] [num_timestep] [is_bias]"
  num_v_in = int(sys.argv[2])
  num_v_out = int(sys.argv[3])
  num_ts = int(sys.argv[4])
  is_bias = int(sys.argv[5])

  test_driver = linear_layer_driver(num_v_in, num_v_out, num_ts, is_bias)
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

if __name__ == '__main__':
  test_name = sys.argv[1]
  supported_test = ('lstm', 'linear_layer', 'pooling')
  assert test_name in supported_test, \
    '{} is not supported, supported test is {}'.format(test_name, supported_test)
  
  if test_name == 'lstm':
    test_lstm()
  if test_name == 'linear_layer':
    test_linear_layer()
  if test_name == 'pooling':
    test_pooling()
