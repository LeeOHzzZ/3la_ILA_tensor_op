import json
import sys
import numpy as np
import subprocess
import os

from src.utils import tool as Tool
from src.driverfactory import *

tool = Tool()

def store_tensor(driver_ctx, tensor, n_ts, n_v_in):
  lctx, actx = driver_ctx
  tensor = lctx[tensor.name]
  n_ts = lctx[n_ts.name]
  n_v_in = lctx[n_v_in.name]

  tensor, adpf_bias = tool.get_adpfloat_bias(tensor)
  adpf_bias = int(adpf_bias + 10)

  print('\n--------------------------------------------------------------')
  print('\tinvoking float to adpfloat converter')
  print('--------------------------------------------------------------\n')

  tensor.tofile('./test/inp_q.tmp', sep='\n')
  tool.call_float_adpt_v_cvtr('./test/inp_q.tmp', adpf_bias, './test/inp_q_av.tmp')

  tensor_asm = []
  for i in range(n_ts):
    tensor_asm.append({
      'name' : 'store_act',
      'timestep_idx' : 'ts_' + str(i),
      'idx' : i
    })

  data_lib = {
    'gb_num_vector_in' : n_v_in,
    'gb_num_vector_out' : n_v_in,
    'adpbias_inp' : adpf_bias
  }
  with open('./test/inp_q_av.tmp') as fin:
    inp_v_list = fin.read().splitlines()
  assert len(inp_v_list) == n_v_in * n_ts
  for i in range(n_ts):
    data_lib = tool.vector_to_data_lib(
        inp_v_list[i*n_v_in : (i+1)*n_v_in], f'ts_{i}', n_v_in, data_lib)

  new_actx = {'tensor': DataBlock(
    'gb_core_store_large',
    0,
    tensor.size,
    {'n_vec': n_v_in, 'n_ts': n_ts, 
     'adpf_bias': adpf_bias,
     'orig_': tensor  # for comparing
    }
  )}

  return ((tensor_asm, data_lib), new_actx)


def load_tensor(driver_ctx, dest):
  _, actx = driver_ctx
  n_ts = actx['tensor'].type_info['n_ts']

  tensor_asm = []
  for i in range(n_ts):
    tensor_asm.append({
      'name' : 'load_act',
      'mem_idx' : 0,
      'ts_idx' : i
    })

  def callback(sim_result, drctx):
    lctx, actx = drctx
    n_ts = actx['tensor'].type_info['n_ts']
    n_vec = actx['tensor'].type_info['n_vec']
    bias = actx['tensor'].type_info['adpf_bias']

    tool.collect_axi_out(in_path = sim_result, 
                    out_path = './test/result.tmp',
                    mem_idx = 0, 
                    num_ts = n_ts, 
                    num_vi = n_vec,
                    num_vo = n_vec, 
                    bias = bias,
                    dtype = 'float32',
                    mem_type='large')
    
    lctx[dest.name] = \
      np.fromfile("./test/result.tmp", sep='\n').astype('float32')
    
    lctx['tensor_orig_'] = actx['tensor'].type_info['orig_']

    return (lctx, actx)
  
  return (tensor_asm, {}), callback


def maxpool(driver_ctx):
  lctx, actx = driver_ctx
  n_ts = actx['tensor'].type_info['n_ts']
  n_vec = actx['tensor'].type_info['n_vec']
  bias = actx['tensor'].type_info['adpf_bias']

  actx['tensor'].type_info['n_ts'] = n_ts // 2

  return ([{
    'name': 'maxp',
    'num_ts': n_ts
  }], {}), actx


simple_pooling_mem = MemoryModel(lambda: {}, [
  MemInstruction('store_tensor', 
    [('tensor', LocalRef), ('n_ts', LocalRef), 
     ('n_v_in', LocalRef)],
    store_tensor),
  MemInstruction('load_tensor', 
    [('dest', LocalRef)], load_tensor)
])

ASMDriver = build_driver(AcceleratorASM(
  simple_pooling_mem, [
  ASMInstruction('maxpool', [], maxpool)
]), namespace='fnlp')


def produce_ref_result(inp, n_ts, n_v_in, repeats):
    """
    reshape the input data to each timestep
    """
    inp = inp.reshape((n_ts, 16*n_v_in))
    out = []
    for i in range(n_ts >> repeats):
      windows = inp[i << repeats:(i+1) << repeats, ]
      out.append(np.maximum.reduce(windows))
    return out


if __name__ == "__main__":
  assert len(sys.argv) in [3, 4], \
    "Usage: python3 asm_driver.py <num_vector_in> <num_timestep> [num_repeats]"
  
  subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
  
  num_v_in = int(sys.argv[1])
  num_ts = int(sys.argv[2])
  repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 1
  
  print('\n--------------------------------------------------------------')
  print('\tproducing random input data')
  print('--------------------------------------------------------------\n')
  coef = 1

  tensor = \
    coef * np.random.uniform(-1, 1, (num_ts * 16 * num_v_in)).astype(np.float32)

  program = [
    {
      'name': 'fnlp.store_tensor',
      'tensor': 'tensor',
      'n_ts': 'n_ts',
      'n_v_in': 'n_v_in'
    }
  ] + [{'name': 'fnlp.maxpool'}] * repeats + [
    {
      'name': 'fnlp.load_tensor',
      'dest': 'result'
    }
  ]

  print(program)

  test_driver = ASMDriver(program, {
    'tensor': tensor,
    'n_ts': num_ts,
    'n_v_in': num_v_in
  })

  test_driver.try_finish() # should finish

  result = test_driver.local_ctx()['result']
  ref_result = produce_ref_result(
    test_driver.local_ctx()['tensor_orig_'],
    num_ts, num_v_in, repeats
  )

  # print(result)

  print('\n--------------------------------------------------------------')
  print('\tanalyze ILA simulation result')
  print('--------------------------------------------------------------\n')
  err_ref_list = []
  ts_stdd_list = []
  for i in range(num_ts >> repeats):
    if not os.environ.get('USE_3LA_FPGA'):
      result_ts = result[num_v_in*16*i : num_v_in*16*(i+1)]
    ref = tool.get_adpfloat_bias(ref_result[i])[0]
    err_out, err_ref = tool.cal_error(result_ts, ref)
    print("result timestep No.{} --- relative error (vs. sim_out): {:5.5%}\
          relative error (vs. ref): {:5.5%}\n".format(i, err_out, err_ref))
    # if is_verbose:
    #   print("reference output: \n{}\nresult: \n{}\n".format(ref, result_ts))
    # err_ref_list.append(err_ref)

    avg_mm, ts_stdd = tool.cal_error_single_tensor(result_ts, ref)
    err_ref_list.append(avg_mm)
    ts_stdd_list.append(ts_stdd)
  
  # clean up
  for file in os.listdir('./test'):
    if '.tmp' in file:
      subprocess.run(['rm', './test/'+file])

