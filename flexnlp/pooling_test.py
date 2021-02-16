import sys

from pooling_driver import pooling_layer_driver as driver

if __name__ == '__main__':
  assert len(sys.argv) == 4, \
    "Usage: python3 pooling_test.py [mode] [num_vector_in] [num_timestep]"

  mode = sys.argv[1]
  num_v_in = int(sys.argv[2])
  num_ts = int(sys.argv[3])


  test_driver = driver(mode, num_v_in, num_ts)
  verbose_analysis = 0
  test_driver.run_test(verbose_analysis)
  test_driver.clean_up()