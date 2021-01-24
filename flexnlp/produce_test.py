import json
import sys
import subprocess

import numpy as np
sys.path.append('./tool/')
from adaptivfloat import quantize_floatext as quantize_floatext

def produce_asm(out_path):
  ila_asm = []
  # some store instructions
  # for i in range(4):
  #   ila_asm.append({
  #     'name' : 'store_act',
  #     'timestep_idx' : 'ts_' + str(i),
  #     'idx' : i
  #   })
  # maxp instruction
  # ila_asm.append({
  #   'name' : 'maxp',
  #   'num_ts' : 4
  # })
  ila_asm.append({
    'name' : 'linear_layer',
    'num_ts' : 10,
    'is_bias' : 1
  })

  ila_asm.append({
    'name' : 'load_act',
    'mem_idx' : 1,
    'ts_idx' : 1
  })

  # ila_asm.append({
  #   'name' : 'store_wgt',
  #   'wgt_idx' : 'w0'
  # })

  # ila_asm.append({
  #   'name' : 'store_bias',
  #   'bias_idx' : 'b0'
  # })

  
  ila_asm = {'asm': ila_asm}
  with open(out_path, 'w') as f:
    json.dump(ila_asm, f, indent=4)
  
  print("flexnlp-ila tensor assembly has been dumped to " + out_path)

def produce_data(out_path):
  data_lib = {}
  
  # # set up the num_vector
  # data_lib['gb_num_vector_in'] = 4
  data_lib['gb_num_vector_in'] = 16
  data_lib['gb_num_vector_out'] = 4
  data_lib['adpbias_wgt'] = 3
  data_lib['adpbias_bias'] = 3
  data_lib['adpbias_inp'] = 3
  data_lib['adpbias_pe_act'] = 4
  # set up wgt tile number
  # data_lib['w0_num_tile'] = 32

  # set up some random data
  data_lib['ts_0.0'] = '0x3C5ACB7A2CC234751CA3281B0231DB4'
  data_lib['ts_0.1'] = '0x0E4011C1A9032FBA813D3DC01A38A9AD'
  data_lib['ts_0.2'] = '0x0A2C2B4B5A9A04AAC3FD99843ABB93896'
  data_lib['ts_0.3'] = '0x37C9A3B512001ABE1431B59601B7BC96'

  data_lib['ts_1.0'] = '0x24D3B0D1B2CB33505FC4178149004ACB'
  data_lib['ts_1.1'] = '0x17404EC79A0A32C81C3F46A2284281B4'
  data_lib['ts_1.2'] = '0x0ACD1B5C0C5B64E1043E2B349B4B74727'
  data_lib['ts_1.3'] = '0x4CD199D31A9743DB2C87C08B01C7B8B0'

  data_lib['ts_2.0'] = '0x2FDDAFD9BEC93A5467B7A58159A354D3'
  data_lib['ts_2.1'] = '0x1B3756C91A1136CD28454DAA324633B5'
  data_lib['ts_2.2'] = '0x0B1D8AFB0D2C0523A46E9BE4EB4A1513C'
  data_lib['ts_2.3'] = '0x53D394D9200159E33335C3AF01CFA5C0'

  data_lib['ts_3.0'] = '0x34E2A1DBC6C132566DBCBC005DB04AD6'
  data_lib['ts_3.1'] = '0x295CC8351739D1244A52B8374CB3B5'
  data_lib['ts_3.2'] = '0x0B5DEA101D8B753454EF2C351AC185643'
  data_lib['ts_3.3'] = '0x54D18ADB1C2664E13B4BC5BB01CB00C4'


  with open(out_path, 'w') as f:
    json.dump(data_lib, f, indent=4)

  print('sample data file has been dumped to ' + out_path)


# -------------------------------------------------------------
# Linear Layer test
# -------------------------------------------------------------
def produce_linear_layer_asm(num_ts, is_bias):
  ila_asm = []

  ila_asm.append({
    'name' : 'store_wgt',
    'wgt_idx' : 'w0'
  })

  ila_asm.append({
    'name' : 'store_bias',
    'bias_idx' : 'b0'
  })

  for i in range(num_ts):
    ila_asm.append({
      'name' : 'store_act',
      'timestep_idx' : 'ts_' + str(i),
      'idx' : i
    })
    
  ila_asm.append({
    'name' : 'linear_layer',
    'num_ts' : num_ts,
    'is_bias' : is_bias
  })
  
  ila_asm = {'asm': ila_asm}

  with open('./test/ly_asm.json', 'w') as fout:
    json.dump(ila_asm, fout, indent=4)
  
  print("linear_layer asm file has been dumped to ./test/ly_asm.json")
  

def wgt_tiling(wgt_in, num_vector_in, num_vector_out):
  ret = np.zeros((1,16))
  for i in range(num_vector_out):
    for j in range(num_vector_in):
      t = wgt_in[16*i:16*(i+1), 16*j:16*(j+1)]
      ret = np.concatenate((ret, t), axis=0)
  return ret[1:,]

def exec_converter(file_in, bias, file_out):
  cmd_0 = ['./npy/adpfloat_converter.out', file_in, str(bias), file_out]
  cmd_1 = ['rm', '-f', file_in]
  subprocess.run(cmd_0)
  subprocess.run(cmd_1)

def produce_data_lib_ly(param, num_ts, out_path):
  """
  Produce data_lib for linear layer
  """
  data_lib = param
  # data_lib = {}

  num_v_in = param['gb_num_vector_in']
  num_v_out = param['gb_num_vector_out']

  # write wgt data into data_lib
  with open('./npy/wgt_qt_v', 'r') as fin:
    wgt_v_list = fin.read().splitlines()
  assert len(wgt_v_list)%16 == 0
  for t in range(num_v_in * num_v_out):
    for v in range(16):
      data_lib['w0.t'+str(t)+'.'+str(v)] = wgt_v_list[16*t+v] 

  # write bias data into data_lib
  with open('./npy/bias_q_v', 'r') as fin:
    bias_v_list = fin.read().splitlines()
  assert len(bias_v_list) == num_v_out
  for v in range(num_v_out):
    data_lib['b0.' + str(v)] = bias_v_list[v]

  # write inp data into data_lib
  with open('./npy/inp_q_v', 'r') as fin:
    inp_v_list = fin.read().splitlines()
  assert len(inp_v_list) == num_v_in * num_ts
  for ts_idx in range(num_ts):
    for v in range(num_v_in):
      data_lib['ts_'+str(ts_idx)+'.' + str(v)] = inp_v_list[v]

  
  with open(out_path, 'w') as fout:
    json.dump(data_lib, fout, indent=4)
  
  print("linear_layer test data_lib has been dumped to " + out_path)


def produce_linear_layer_test_data(num_vector_in, num_vector_out, num_ts):
  wgt_init = np.random.random_sample((16*num_vector_out, 16*num_vector_in))
  inp_init = np.random.random_sample((num_vector_in * 16 * num_ts))
  bias_init = np.random.random_sample((num_vector_out*16))

  wgt_q, bias_wgt = quantize_floatext(wgt_init)
  inp_q, bias_inp = quantize_floatext(inp_init)
  bias_q, bias_b = quantize_floatext(bias_init)

  wgt_qt = wgt_tiling(wgt_q, num_vector_in, num_vector_out)

  bias_wgt += 10 
  bias_inp += 10
  bias_b += 10

  param = {
    'gb_num_vector_in' : num_vector_in,
    'gb_num_vector_out' : num_vector_out,
    'adpbias_wgt' : int(bias_wgt),
    'adpbias_inp' : int(bias_inp),
    'adpbias_bias' : int(bias_b),
    'adpbias_pe_act' : 2,
    'w0_num_tile' : int(num_vector_in * num_vector_out)
  }

  wgt_qt.tofile('./npy/wgt_qt', sep = '\n')
  inp_q.tofile('./npy/inp_q', sep = '\n')
  bias_q.tofile('./npy/bias_q', sep = '\n')
  # print("(bias_wgt, bias_inp, bias_b) is ", (bias_wgt, bias_inp, bias_b))
  # call converter binary to do the conversion
  exec_converter("./npy/wgt_qt", bias_wgt, "./npy/wgt_qt_v")
  exec_converter("./npy/inp_q", bias_inp, "./npy/inp_q_v")
  exec_converter("./npy/bias_q", bias_b, "./npy/bias_q_v")

  produce_data_lib_ly(param, num_ts, './test/ly_data_lib.json')

def produce_linear_layer_test(num_vector_in, num_vector_out, num_ts, is_bias):
  produce_linear_layer_asm(num_ts, is_bias)
  produce_linear_layer_test_data(num_vector_in, num_vector_out, num_ts)

# ===============================================
# ===============================================
if __name__ == "__main__":
  asm_out_path = 'asm_test.json'
  data_out_path = 'data_lib_test.json'
  if len(sys.argv) > 2:
    asm_out_path = sys.argv[1]
    data_out_path = sys.argv[2]
  
  # produce_asm(asm_out_path)
  # produce_data(data_out_path)
  produce_linear_layer_test(16, 4, 10, 1)
