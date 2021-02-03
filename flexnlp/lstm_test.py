import sys

from lstm_driver import lstm_layer_driver as driver

if __name__ == '__main__':
  assert len(sys.argv) == 6, \
    "Usage: python3 lstm_test.py [num_vector_in] [num_vector_out] [num_timestep] [is_bias] [is_zero_first]"

  num_v_in = int(sys.argv[1])
  num_v_out = int(sys.argv[2])
  num_ts = int(sys.argv[3])
  is_bias = int(sys.argv[4])
  is_zero_first = int(sys.argv[5])

  test_driver = driver(num_v_in, num_v_out, num_ts, is_bias, is_zero_first)

  test_driver.produce_lstm_asm()
  test_driver.produce_random_test_data()
  test_driver.produce_lstm_data_lib()
  test_driver.gen_prog_frag()
  test_driver.invoke_ila_simulator()
  test_driver.get_ila_sim_result()
  test_driver.gen_axi_cmds()
  test_driver.produce_ref_result()
  test_driver.result_analysis(1)