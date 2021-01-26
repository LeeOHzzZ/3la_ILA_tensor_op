import json
import sys
import subprocess
import numpy as np

from produce_test import produce_linear_layer_test
from ts_to_asm_converter import get_gb_large_addr_offset
from ts_to_asm_converter import get_gb_base_addr_1

def cal_error(result, ref):
  diff = result - ref
  abs_diff = np.abs(diff)
  mean_diff = np.sum(abs_diff) / (result.size * ref.size)
  return mean_diff/np.mean(result), mean_diff/np.mean(ref)

def axi_out2float(in_path, out_path, mem_idx, num_ts, num_v_in, num_v_out, bias):
  with open(in_path, 'r') as fin:
    v_data = json.load(fin)
  data_list = []
  mem_base = get_gb_base_addr_1(num_ts, num_v_in)*16 # byte level

  for ts_idx in range(num_ts):
    for v_idx in range(num_v_out):
      addr = mem_base + get_gb_large_addr_offset(ts_idx, num_v_out, v_idx)
      addr_str = '0x{:08X}'.format(addr)
      data_str = v_data[addr_str][2:]
      assert len(data_str) == 32, "wrong length for ILA simulator return result"
      for b_idx in range(16):
        data_list.append('0x{}\n'.format(data_str[30-2*b_idx:32-2*b_idx]))
  with open('./npy/ila_result_temp', 'w') as fout:
    fout.writelines(data_list)
  subprocess.run(['./npy/adpfloat_to_float.out',
                  './npy/ila_result_temp',
                  str(bias),
                  out_path])

if __name__ == '__main__':
  assert len(sys.argv) == 5, \
    "Usage: python3 linear_layer_testflow.py [num_vector_in] [num_vector_out] [num_timestep] [is_bias]"
  num_v_in = int(sys.argv[1])
  num_v_out = int(sys.argv[2])
  num_ts = int(sys.argv[3])
  is_bias = int(sys.argv[4])

  bias_act = produce_linear_layer_test(num_v_in, num_v_out, num_ts, is_bias)

  print('\n--------------------------------------------------------------')
  print('\tgenerate prog_frag.json for ILA simulator')
  print('--------------------------------------------------------------\n')

  subprocess.run(['python3', 
                  'gen_prog_frag.py', 
                  './test/ly_asm.json', 
                  './test/ly_data_lib.json', 
                  './test/ly_prog_frag_in.json'])

  print('\n--------------------------------------------------------------')
  print('\tinvoking ILA simulator')
  print('--------------------------------------------------------------\n')

  subprocess.run(['./test/asm_sim_driver.out',
                  './test/ly_prog_frag_in.json',
                  './test/ly_adpf_result.txt'])

  print('\n--------------------------------------------------------------')
  print('\tcollecting ILA simulation result')
  print('--------------------------------------------------------------\n')

  axi_out2float('./test/ly_adpf_result.txt', \
                './test/ly_float_result.txt', \
                1, num_ts, num_v_in, num_v_out, bias_act)

  result = np.fromfile('./test/ly_float_result.txt', sep = '\n')
  print("result's shape: {}\n".format(result.shape))

  for i in range(num_ts):
    result_ts = result[num_v_out*16*i : num_v_out*16*(i+1)]
    ref = np.fromfile('./npy/ref_' + str(i) + '.txt', sep = '\n')
    err_out, err_ref = cal_error(result_ts, ref)
    print("result timestep No.{} --- relative error (vs. sim_out): {:5.5%}\
           relative error (vs. ref): {:5.5%}".format(i, err_out, err_ref))
    print(result_ts)
    print('\n')
