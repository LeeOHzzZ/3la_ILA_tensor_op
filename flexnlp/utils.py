"""
This file contains tools for the testflow
"""
import json
import sys
import subprocess
import numpy as np

sys.path.append('./tool')
import tvm
from tvm.contrib.ly3la.flexnlp.tool.adaptivfloat import quantize_floatext
from tvm.contrib.ly3la.flexnlp.tool.relay_lstm import relay_lstm_ref
from tvm.contrib.ly3la.flexnlp.tool.relay_layers import relay_layernorm
# from adaptivfloat import quantize_floatext
# from relay_lstm import relay_lstm_ref
# from relay_layers import relay_layernorm

FLEXNLP_VECTOR_SIZE = 16
FLEXNLP_GBCORE_NUM_BANKS = 16

class tool:
  def __init__(self):
    pass
  
  def cal_error(self, result, ref):
    diff = result - ref
    abs_diff = np.abs(diff)
    mean_diff = np.sum(abs_diff) / (diff.size)
    # print(result.size, ref.size)
    return mean_diff/np.mean(np.abs(result)), mean_diff/np.mean(np.abs(ref))

  def get_adpfloat_bias(self, array):
    """
    return the quantized matrix and adpfloat bias
    """
    return quantize_floatext(array)
  
  def get_relay_lstm_ref(self, num_v_in, num_v_out, num_ts,
                         inp, wgt_i, wgt_h, bias_i, bias_h):
    """
    return relay lstm reference data
    """
    return relay_lstm_ref(num_v_in, num_v_out, num_ts,
                          inp, wgt_i, wgt_h, bias_i, bias_h)
  
  def get_relay_layernorm_ref(self, num_v, inp, beta, gamma):
    """
    return relay layernorm reference data
    """
    return relay_layernorm(num_v, inp, beta, gamma)
  
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
  
  def lstm_wgt_tiling(self, wgt_in, num_v_in, num_v_out):
    """
    This function performs weight matrixing tiling for LSTM
    """
    ret = np.zeros((1,16))
    assert num_v_out % 4 == 0, 'wrong value for num_v_out'
    assert wgt_in.shape == (16*4*num_v_out, 16*num_v_in), \
      'lstm_wgt_tiling: wgt_in shape {} doesn\'t match {}'.format(wgt_in.shape, (16*4*num_v_out, 16*num_v_in))
    sub_height = 16*num_v_out
    W_i = wgt_in[0 : sub_height, ]
    W_f = wgt_in[sub_height : 2*sub_height, ]
    W_g = wgt_in[2*sub_height : 3*sub_height, ]
    W_o = wgt_in[3*sub_height : 4*sub_height, ]
    
    num_out_pe = num_v_out >> 2
    sub_height_pe = sub_height >> 2 # sub height for matrice in each PE
    for pe_idx in range(4):
      # first divide each matrix into 4 parts for 4 PEs
      W_i_p = W_i[pe_idx*sub_height_pe : (pe_idx+1)*sub_height_pe]
      W_f_p = W_f[pe_idx*sub_height_pe : (pe_idx+1)*sub_height_pe]
      W_g_p = W_g[pe_idx*sub_height_pe : (pe_idx+1)*sub_height_pe]
      W_o_p = W_o[pe_idx*sub_height_pe : (pe_idx+1)*sub_height_pe]
      for out_idx in range(num_out_pe):
        # slicing each matrix for different output vectors
        # tiling W_i
        for in_idx in range(num_v_in):
          t_i = W_i_p[16*out_idx : 16*(out_idx+1), 16*in_idx : 16*(in_idx+1)]
          ret = np.concatenate((ret, t_i), axis=0)
        # tiling W_g
        for in_idx in range(num_v_in):
          t_g = W_g_p[16*out_idx : 16*(out_idx+1), 16*in_idx : 16*(in_idx+1)]
          ret = np.concatenate((ret, t_g), axis=0)
        # tiling W_f
        for in_idx in range(num_v_in):
          t_f = W_f_p[16*out_idx : 16*(out_idx+1), 16*in_idx : 16*(in_idx+1)]
          ret = np.concatenate((ret, t_f), axis=0)
        # tiling W_o
        for in_idx in range(num_v_in):
          t_o = W_o_p[16*out_idx : 16*(out_idx+1), 16*in_idx : 16*(in_idx+1)]
          ret = np.concatenate((ret, t_o), axis=0)

    return ret[1:]
  
  def lstm_bias_reshape(self, bias_in, num_v):
    """
    This function reshape bias vectors for flexnlp LSTM op
    """
    # flexnlp bias arrangement is (b_i, b_g, b_f, b_o)
    assert bias_in.shape == (4*16*num_v, ), \
      'lstm_bias_reshape: bias shape should be {} instead of {}'.format((4*16*num_v, ), bias_in.shape)
    bias_in = bias_in.reshape((4, 16*num_v))
    B_i = bias_in[0, ]
    B_f = bias_in[1, ]
    B_g = bias_in[2, ]
    B_o = bias_in[3, ]

    num_v_out_pe = num_v >> 2
    ret = []

    for pe_idx in range(4):
      for out_idx in range(num_v_out_pe):
        idx_head = (pe_idx*num_v_out_pe + out_idx)*16
        idx_end = idx_head + 16
        ret.append(B_i[idx_head : idx_end])
        ret.append(B_g[idx_head : idx_end])
        ret.append(B_f[idx_head : idx_end])
        ret.append(B_o[idx_head : idx_end])
    return np.concatenate(ret)
    
  
  def wgt_to_data_lib(self, wgt, wgt_id, num_tile, data_lib):
    """
    Write weight data into data_lib
    """
    assert len(wgt) % 16 == 0
    for t in range(num_tile):
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
    cmd_0 = ['float_to_adpfloat.out', in_path, str(bias), out_path]
    cmd_1 = ['rm', '-f', in_path]
    subprocess.run(cmd_0)
    subprocess.run(cmd_1)
  
  def call_ila_simulator(self, in_path, out_path):
    """
    call pre-built flexnlp-ila simulator to execute the ila program fragment
    """
    subprocess.run(['asm_sim_driver.out',
                    in_path, out_path])

  def call_adpt_float_cvtr(self, in_path, bias, out_path):
    """
    call pre-built binary to convert adaptive-float data into float data
    """
    subprocess.run(['adpfloat_to_float.out',
                    in_path, str(bias), out_path])

  def axi_out_to_float(self, in_path, out_path, mem_idx, num_ts, num_vi, num_vo, bias):
    """
    convert the axi read return to floating point data
    """
    with open(in_path, 'r') as fin:
      v_data = json.load(fin)
    data_list = []
    # mem_base = self.get_gb_base_addr_1(num_ts, num_vi)*16 # byte level address
    mem_base = 0 if mem_idx == 0 else self.get_gb_base_addr_1(num_ts, num_vi)*16

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
  
  def get_pe_base_bias_v_1(self, num_v):
    # get bias base address in pe input buffer (vector level)
    return num_v + 0x10
  
  def get_pe_base_bias_v_2(self, num_v_in, num_v_out):
    # get bias base address in pe input buffer for hidden state
    return self.get_pe_base_bias_v_1(num_v_in) + num_v_out*2 + 0x20

  