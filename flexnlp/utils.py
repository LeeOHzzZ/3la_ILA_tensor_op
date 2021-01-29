"""
This file contains tools for the testflow
"""
import json
import sys
import subprocess
import numpy as np

sys.path.append('./tool')
from adaptivfloat import quantize_floatext

FLEXNLP_VECTOR_SIZE = 16
FLEXNLP_GBCORE_NUM_BANKS = 16

class tool:
  def __init__(self):
    # self.num_vector_in = config['num_vector_in']
    # self.num_vector_out = config['num_vector_out']
    # self.
    pass
  
  def cal_error(self, result, ref):
    diff = result - ref
    abs_diff = np.abs(diff)
    mean_diff = np.sum(abs_diff) / (result.size * ref.size)
    return mean_diff/np.mean(result), mean_diff/np.mean(ref)

  def get_adpfloat_bias(self, array):
    """
    return the quantized matrix and adpfloat bias
    """
    return quantize_floatext(array)
  
  def wgt_tiling(self, wgt_in, num_v_in, num_v_out):
    """
    This function takes a numpy array input, perform 16x16
    tiling of the matrix, and return a 1-dimension array
    """
    ret = np.zeros((1,16))
    for i in range(num_v_out):
      for j in range(num_v_in):
        t = wgt_in[16*i:16*(i+1), 16*j:16*(j+1)]
        ret = np.concatenate((ret, t), axis=0)
    return ret[1:,]
  
  def wgt_to_data_lib(self, wgt, wgt_id, num_v_in, num_v_out, data_lib):
    """
    Write weight data into data_lib
    """
    assert len(wgt) % 16 == 0
    for t in range(num_v_in * num_v_out):
      for v in range(16):
        data_lib['{}.t{}.{}'.format(wgt_id, t, v)] = wgt[16*t+v]
    return data_lib
  
  def vector_to_data_lib(self, v_list, id, num_v, data_lib):
    """
    write vector data into data_lib (input timestep and bias etc.)
    """
    assert len(v_list) == num_v
    for v in range(num_v):
      data_lib['{}.{}'.format(id, v)] = v_list[v]
    return data_lib
  
  def call_float_adpt_v_cvtr(self, in_path, bias, out_path):
    """
    call pre-built binary to convert float32 data into 
    adaptive-float 16byte vector data
    """
    cmd_0 = ['./tool/float_to_adpfloat.out', in_path, str(bias), out_path]
    cmd_1 = ['rm', '-f', in_path]
    subprocess.run(cmd_0)
    subprocess.run(cmd_1)
  
  def call_ila_simulator(self, in_path, out_path):
    """
    call pre-built flexnlp-ila simulator to execute the ila program fragment
    """
    subprocess.run(['./tool/asm_sim_driver.out',
                    in_path, out_path])

  def call_adpt_float_cvtr(self, in_path, bias, out_path):
    """
    call pre-built binary to convert adaptive-float data into float data
    """
    subprocess.run(['./tool/adpfloat_to_float.out',
                    in_path, str(bias), out_path])

  def axi_out_to_float(self, in_path, out_path, mem_idx, num_ts, num_vi, num_vo, bias):
    """
    convert the axi read return to floating point data
    """
    with open(in_path, 'r') as fin:
      v_data = json.load(fin)
    data_list = []
    mem_base = self.get_gb_base_addr_1(num_ts, num_vi)*16 # byte level address

    for ts_idx in range(num_ts):
      for v_idx in range(num_vo):
        addr = mem_base + self.get_gb_large_addr_offset(ts_idx, num_vo, v_idx)
        addr_str = '0x{:08X}'.format(addr)
        data_str = v_data[addr_str][2:]
        assert len(data_str) == 32, "wrong length for ILA simulator return result"
        for b_idx in range(16):
          data_list.append('0x{}\n'.format(data_str[30-2*b_idx:32-2*b_idx]))

    with open('./test/ila_result.tmp', 'w') as fout:
      fout.writelines(data_list)

    self.call_adpt_float_cvtr('./test/ila_result.tmp', bias, out_path)        
  

  def get_gb_large_addr_offset(self, ts_idx, num_vector, vector_idx):
    # calculate the start address offset in the gbcore large buffer
    timestep_size = num_vector * FLEXNLP_VECTOR_SIZE
    group_size = timestep_size * FLEXNLP_GBCORE_NUM_BANKS
    gb_buf_row_size = FLEXNLP_GBCORE_NUM_BANKS * FLEXNLP_VECTOR_SIZE
    out = (int(ts_idx/FLEXNLP_GBCORE_NUM_BANKS)) * group_size + \
          (ts_idx%FLEXNLP_GBCORE_NUM_BANKS) * FLEXNLP_VECTOR_SIZE + \
          gb_buf_row_size * vector_idx
    return out

  def get_gb_base_addr_1(self, num_ts, num_v_in):
    # get base address for gb large buffer of memory index 1 (vector_level)
    return int(num_ts/16 + 2) * 16 * num_v_in

  