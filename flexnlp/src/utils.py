"""
This file contains tools for the testflow
"""
import json
import sys
import subprocess
import numpy as np
import timeit
import os

from math import sqrt

# sys.path.append('./tool')
from .tool.adaptivfloat import quantize_floatext
from .tool.relay_lstm import relay_lstm_ref
from .tool.relay_layers import relay_layernorm
from .tool.relay_layers import relay_attention

FLEXNLP_VECTOR_SIZE = 16
FLEXNLP_GBCORE_NUM_BANKS = 16
FLEXNLP_ADDR_BASE = 0x33000000
FLEXNLP_PE_PARTITION_OFFSET = 0x01000000
FLEXNLP_PE_ACT_BUF_BASE = 0x00900000
FLEXNLP_GB_LARGE_BUF_BASE = FLEXNLP_ADDR_BASE + 0x00500000
FLEXNLP_GB_SMALL_BUF_BASE = FLEXNLP_ADDR_BASE + 0X00600000


class tool:
  def __init__(self):
    pass
  
  def cal_error(self, result, ref):
    diff = np.abs(result - ref)
    mean_diff = np.mean(diff)
    # return np.mean(diff/np.abs(result)), np.mean(diff/np.abs(ref))
    return mean_diff/np.mean(np.abs(result)), mean_diff/np.mean(np.abs(ref))
  

  def cal_error_single_tensor(self, result, ref):
    """
    This function calculate the mean error and standard deviation among the elements
    a single result tensor compared with the reference tensor

    Calculate the Frobenius Norm or 2-Norm of the tensors and return the relative errors
    """
    # diff = np.abs(result - ref)
    # avg_mismatch = np.mean(diff) / np.mean(np.abs(ref))
    # stdd = np.std(diff) / np.mean(np.abs(ref))
    diff = result - ref
    # relative mis-match
    rmm = np.linalg.norm(diff)/np.linalg.norm(ref)
    return rmm


  def cal_mean_stdd(self, data_list):
    """
    This function calculate the mean and standard deviation of the input data list
    """
    # mean = sum(data_list) / len(data_list)
    # stdd = sqrt(sum(list(map(lambda x: (x - mean)**2, data_list)))/len(data_list))
    data_list_tensor = np.array(data_list)
    mean = np.mean(data_list_tensor)
    stdd = np.std(data_list_tensor)
    return mean, stdd


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
  
  def get_relay_attention(self, key_seq_len, query_seq_len, vector_size, 
                          enc_data, dec_data, wgt_data):
    """
    return relay attention reference data
    """
    return relay_attention(key_seq_len, query_seq_len, vector_size, 
                            enc_data, dec_data, wgt_data)
  
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
    cmd_0 = ['float_to_adpfloat', in_path, str(bias), out_path]
    cmd_1 = ['rm', '-f', in_path]
    subprocess.run(cmd_0)
    subprocess.run(cmd_1)

  def call_pack_int8_to_vector(self, in_path, out_path):
    """
    call pre-built binary to pack int8 data into vector string 
    for FlexASR
    """
    cmd_0 = ['pack_int8_to_vector', in_path, out_path]
    cmd_1 = ['rm', '-f', in_path]
    subprocess.run(cmd_0)
    subprocess.run(cmd_1)
  
  def call_ila_simulator(self, dtype, in_path, out_path):
    """
    call pre-built flexnlp-ila simulator to execute the ila program fragment
    """
    if dtype == "float32":
      print("\n[FLEXASR_ILA] calling adptfloat-version ila simulator\n")
      subprocess.run(['flex_asm_sim_driver', in_path, out_path])
    elif dtype == "int8":
      print("\n[FLEXASR_ILA] calling int8-version ila simulator\n")
      subprocess.run(['flex_asm_sim_driver_int8', in_path, out_path])

  def call_adpt_float_cvtr(self, in_path, bias, out_path):
    """
    call pre-built binary to convert adaptive-float data into float data
    """
    subprocess.run(['adpfloat_to_float',
                    in_path, str(bias), out_path])

  def call_collect_int8_result(self, in_path, out_path):
    """
    call pre-built binary to collect int8 results
    """
    subprocess.run(["collect_int8_result", in_path, out_path])

  def collect_axi_out(self, in_path, out_path, mem_idx, num_ts, num_vi, num_vo, bias,
                       dtype, mem_type):
    """
    convert the axi read return to floating point data
    """
    with open(in_path, 'r') as fin:
      v_data = json.load(fin)
    data_list = []
    # mem_base = self.get_gb_base_addr_1(num_ts, num_vi)*16 # byte level address
    mem_base = 0 if mem_idx == 0 else self.get_gb_base_addr_1(num_ts, num_vi)*16
    
    if mem_type == "large":
      for ts_idx in range(num_ts):
        for v_idx in range(num_vo):
          addr = (
            FLEXNLP_GB_LARGE_BUF_BASE + 
            mem_base + 
            self.get_gb_large_addr_offset(ts_idx, num_vo, v_idx)
          )
          addr_str = '0x{:08X}'.format(addr)
          data_str = v_data[addr_str][2:]
          assert len(data_str) == 32, "wrong length for ILA simulator return result"
          for b_idx in range(16):
            data_list.append('0x{}\n'.format(data_str[30-2*b_idx:32-2*b_idx]))
    elif mem_type == "small":
      for v_idx in range(num_vo):
        addr = FLEXNLP_GB_SMALL_BUF_BASE + mem_base + 16*v_idx
        addr_str = "0x{:08X}".format(addr)
        data_str = v_data[addr_str][2:]
        assert len(data_str) == 32, "wrong length for ILA simulator return result"
        for b_idx in range(16):
          data_list.append('0x{}\n'.format(data_str[30-2*b_idx:32-2*b_idx]))
    elif mem_type == "pe_act":
      for pe_idx in range(4):
        for v_idx in range(num_vo // 4):
          addr = (
            FLEXNLP_ADDR_BASE + 
            pe_idx * FLEXNLP_PE_PARTITION_OFFSET +
            FLEXNLP_PE_ACT_BUF_BASE +
            v_idx * 0x010
          )
          addr_str = "0x{:08X}".format(addr)
          data_str = v_data[addr_str][2:]
          assert len(data_str) == 32, "wrong length for ILA simulator return result"
          for b_idx in range(16):
            data_list.append('0x{}\n'.format(data_str[30-2*b_idx:32-2*b_idx]))
  
    with open('./test/ila_result.tmp', 'w') as fout:
      fout.writelines(data_list)

    if dtype == "float32":
      self.call_adpt_float_cvtr('./test/ila_result.tmp', bias, out_path)
    elif dtype == "int8":
      self.call_collect_int8_result("./test/ila_result.tmp", out_path)

    return np.fromfile(out_path, sep='\n').astype(dtype)
  
  def axi_out_to_float_fpga(self, in_path, out_path, mem_idx, num_ts, num_vi, num_vo, bias,
                            base_addr='0xa0500000'):
    """
    convert the axi read return to floating point data
    TODO: temporally solution to deal with the offset of fpga.
    """
    with open(in_path, 'r') as fin:
      v_data = json.load(fin)
    data_list = []
    # mem_base = self.get_gb_base_addr_1(num_ts, num_vi)*16 # byte level address
    mem_base = 0 if mem_idx == 0 else self.get_gb_base_addr_1(num_ts, num_vi)*16

    for ts_idx in range(num_ts):
      for v_idx in range(num_vo):
        addr = mem_base + self.get_gb_large_addr_offset(ts_idx, num_vo, v_idx) + int(base_addr,16)
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
  
  """
  for invoking ILA simulator 
  """
  def collect_ila_result(self, in_path, mem_idx, num_ts, num_vi, num_vo, bias,
                         dtype = "float32", mem_type = 'large'):
    # mem_type: where result is located in the gb memory
    assert mem_type in ("large", "small", "pe_act")
    print('\n--------------------------------------------------------------')
    print('\tinvoking ILA simulator')
    print('--------------------------------------------------------------\n')
    # measure the time of ila simulation
    start_time = timeit.default_timer()
    self.call_ila_simulator(dtype, in_path, './test/adpf_result.tmp')
    end_time = timeit.default_timer()
    print('\n********* ILA simulator performance ***********')
    print('ILA simulator execution time is {:04f}s'.format(end_time - start_time))
    return self.collect_axi_out(in_path = './test/adpf_result.tmp', 
                                out_path = './test/result.tmp',
                                mem_idx = mem_idx, 
                                num_ts = num_ts, 
                                num_vi = num_vi,
                                num_vo = num_vo, 
                                bias = bias,
                                dtype = dtype,
                                mem_type=mem_type)


  """
  for generating FPGA source code file
  """
  def get_axi_cmd_template(self):
    return (
      '#include <stdio.h>\n'  +
      '#include <string.h>\n'  + 
      '#include "xparameters.h"\n'  +
      '#include "xil_io.h"\n'  +
      '#include "xbasic_types.h"\n'  +
      '#include "arm_neon.h"\n'  +
      '#include "xscugic.h"\n'  +
      '#include "xtime_l.h"\n'  +
      '#include "sleep.h"\n'  +
      'typedef unsigned uint128_t __attribute__ ((mode (TI)));\n'  +

      '#define HW128_REG(ADDRESS)  (*((volatile uint128_t  *)(ADDRESS)))\n'  +
      '#define HW64_REG(ADDRESS)  (*((volatile unsigned long long *)(ADDRESS)))\n'  +
      '#define HW32_REG(ADDRESS)  (*((volatile unsigned int  *)(ADDRESS)))\n'  +
      '#define HW16_REG(ADDRESS)  (*((volatile unsigned short *)(ADDRESS)))\n'  +
      '#define HW8_REG(ADDRESS)   (*((volatile unsigned char  *)(ADDRESS)))\n'  +

      'typedef union {\n'  +
      '  uint128_t val128;\n'  +
      '  int64x2_t val64;\n'  +
      '  int32x4_t val32;\n'  +
      '  int16x8_t val16;\n'  +
      '  int8x16_t val8;\n'  +
      '} smiv128_t;\n'  +

      'smiv128_t weight128;\n'  +
      'smiv128_t read_data;\n' )


  def parse_fpga_results(self, file_in, file_out, op_name):
    """
    this function will parse the results returned by FPGA simulation
    """
    result_dict = {}
    ider = '[read_out_' + op_name + ']:'
    with open(file_in, 'r') as fin:
      raw_in = fin.readlines()
    for l in raw_in:
      if ider not in l:
        continue
      else:
        result_dict.update(json.loads(l[len(ider):]))
    with open(file_out, 'w') as fout:
      json.dump(result_dict, fout, indent=4)
  
  """
  for invoking FPGA simulation
  """
  def collect_fpga_results(self, mem_idx, num_ts, num_vi, num_vo, bias, base_addr, op_name = ''):
    """
    call to FPGA simulation
    """
    # TODO: implement FPGA invoke
    # 1. implement the call cmds to invoke fpga simulation
    # 2. specify the fpga output result path
    # 3. put the output file name in the next function's (collect_fpga_results) argument.
    print('\n--------------------------------------------------------------')
    print('\tcalling FlexNLP FPGA simulation')
    print('--------------------------------------------------------------\n')
    # some example command
    # fpga axi header is at './test/fpga_axi_set_cmds.h'
    cmd_list = ['echo', 'hello_world']
    subprocess.run(cmd_list)
    """
    parse the FPGA simulation results
    """
    print('\n--------------------------------------------------------------')
    print('\tParsing and collect FlexNLP FPGA simulation results')
    print('--------------------------------------------------------------\n')
    self.parse_fpga_results('./test/fpga_output.txt', './test/fpga_adpf_result.tmp', op_name)
    self.axi_out_to_float_fpga('./test/fpga_adpf_result.tmp', './test/fpga_float_result.tmp',
                             mem_idx = mem_idx, num_ts = num_ts, 
                             num_vi = num_vi, num_vo = num_vo, bias=bias, base_addr=base_addr)
    return np.fromfile('./test/fpga_float_result.tmp', sep = '\n')


  def clean_up(self):
    for file in os.listdir('./test'):
      if '.tmp' in file:
        subprocess.run(['rm', './test/'+file])