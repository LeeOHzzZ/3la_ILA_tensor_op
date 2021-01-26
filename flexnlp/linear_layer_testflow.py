import json
import sys
import subprocess
import numpy as np

from produce_test import produce_linear_layer_test

def cal_error(result, ref):
  diff = result - ref
  abs_diff = np.abs(diff)
  mean_diff = np.sum(abs_diff) / (result.size * ref.size)
  return mean_diff/np.mean(result), mean_diff/np.mean(ref)

if __name__ == '__main__':
  assert len(sys.argv) == 5, \
    "Usage: python3 linear_layer_testflow.py [num_vector_in] [num_vector_out] [num_timestep] [is_bias]"
  num_v_in = int(sys.argv[1])
  num_v_out = int(sys.argv[2])
  num_ts = int(sys.argv[3])
  is_bias = int(sys.argv[4])

  produce_linear_layer_test(num_v_in, num_v_out, num_ts, is_bias)

  print('\n-------------------------------------------------')
  print('generate prog_frag.json for ILA simulator')
  print('-------------------------------------------------\n')

  subprocess.run(['python3', 
                  'gen_prog_frag.py', 
                  './test/ly_asm.json', 
                  './test/ly_data_lib.json', 
                  './test/ly_prog_frag_in.json'])

  print('\n-------------------------------------------------')
  print('evoking ILA simulator')
  print('-------------------------------------------------\n')

  subprocess.run(['./test/asm_sim_driver.out',
                  './test/ly_prog_frag_in.json',
                  './test/ly_result.txt'])

  print('\n-------------------------------------------------')
  print('gathering ILA simulation result')
  print('-------------------------------------------------\n')

  result = np.fromfile('./test/ly_result.txt', sep = '\n')
  print("result's shape: {}\n".format(result.shape))

  for i in range(num_ts):
    result_ts = result[num_v_out*16*i : num_v_out*16*(i+1)]
    ref = np.fromfile('./npy/ref_' + str(i) + '.txt', sep = '\n')
    err_out, err_ref = cal_error(result_ts, ref)
    print("result timestep No.{} --- relative error (vs. sim_out): {:5.5%}\
           relative error (vs. ref): {:5.5%}".format(i, err_out, err_ref))
    print(result_ts)
    print('\n')
