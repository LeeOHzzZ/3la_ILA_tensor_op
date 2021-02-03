import json
import sys
import numpy as np
import subprocess
import os

from utils import tool as tool
from converter import Converter as cvtr

class lstm_layer_driver:
  def __init__(self, num_v_in, num_v_out, num_ts, is_bias, is_zero_first):
    self.num_v_in = num_v_in
    self.num_v_out = num_v_out
    self.num_ts = num_ts
    self.is_bias = is_bias
    self.is_zero_first = is_zero_first
    self.tl = tool()

  # ----------------------------
  # producing lstm tensor asm
  # ----------------------------
  def produce_lstm_asm(self):
    ila_asm = []
    ila_asm.append({
      'name' : 'store_wgt_i',
      'wgt_idx' : 'wi'
    })
    ila_asm.append({
      'name' : 'store_wgt_h',
      'wgt_idx' : 'wh'
    })
    ila_asm.append({
      'name' : 'store_bias_i',
      'bias_idx' : 'bi'
    })
    ila_asm.append({
      'name' : 'store_bias_h',
      'bias_idx' : 'bh'
    })
    
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'store_act',
        'timestep_idx' : 'ts_' + str(i),
        'idx' : i
      })
    
    ila_asm.append({
      'name' : 'lstm_layer',
      'num_ts' : self.num_ts,
      'is_bias' : self.is_bias,
      'is_zero_first' : self.is_zero_first
    })
    ila_asm.append({
      'name' : 'wait_irq'
    })
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'load_act',
        'mem_idx' : 1,
        'ts_idx' : i
      })

    ila_asm = {'asm': ila_asm}
    self.ts_asm_lib = ila_asm

    with open('./test/lstm_asm.json', 'w') as fout:
      json.dump(ila_asm, fout, indent=4)
    print('*** ILA tensor assembly has been dumped to ./test/lstm_asm.json ***')

  # ----------------------------
  # produce lstm test data
  # ----------------------------
  def produce_random_test_data(self):
    """
    produce random data for testing
    """
    print('\n--------------------------------------------------------------')
    print('\tproducing random input data')
    print('--------------------------------------------------------------\n')
    coef = 1
    # input weight matrix dimension is (4 x output, input)
    # hidden weight matrix dimension is (4 x output, input)
    wgt_i_init = \
      coef * np.random.uniform(-1, 1, (4*16*self.num_v_out, 16*self.num_v_in)).astype(np.float32)
    wgt_h_init = \
      coef * np.random.uniform(-1, 1, (4*16*self.num_v_out, 16*self.num_v_out)).astype(np.float32)
    inp_init = \
      coef * np.random.uniform(-1, 1, (self.num_v_in * 16 * self.num_ts)).astype(np.float32)
    print('(wgt_i, wgt_h, inp) shape is ({}, {}, {})'.format(
      wgt_i_init.shape, wgt_h_init.shape,
      tuple(t/self.num_ts for t in inp_init.shape)
    ))

    if self.is_bias == 1:
      bias_i_init = \
        coef * np.random.uniform(0, 1, (4*16*self.num_v_out)).astype(np.float32)
      bias_h_init = \
        coef * np.random.uniform(0, 1, (4*16*self.num_v_out)).astype(np.float32)
    else:
      bias_i_init = np.zeros((4*self.num_v_out), dtype = np.float32)
      bias_h_init = np.zeros((4*self.num_v_out), dtype = np.float32)
    
    # initial cell state
    if (self.is_zero_first):
      self.cell_state_init = np.zeros(16*self.num_v_out, dtype = np.float32)
      self.hidden_state_init = np.zeros(16*self.num_v_out, dtype = np.float32)
    else:
      self.cell_state_init = coef * np.random.uniform(-1, 1, (16*self.num_v_out)).astype(np.float32)
      self.hidden_state_init = coef * np.random.uniform(-1, 1, (16*self.num_v_out)).astype(np.float32)
    
    self.wgt_i = wgt_i_init
    self.wgt_h = wgt_h_init
    self.inp = inp_init
    self.bias_i = bias_i_init
    self.bias_h = bias_h_init
  

  def produce_lstm_data_lib(self):
    """
    get quantized inputs, weights and bias
    """
    wgt_i_q, adpbias_wgt_i = self.tl.get_adpfloat_bias(self.wgt_i)
    wgt_h_q, adpbias_wgt_h = self.tl.get_adpfloat_bias(self.wgt_h)
    inp_q, adpbias_inp = self.tl.get_adpfloat_bias(self.inp)
    bias_i_q, adpbias_b_i = self.tl.get_adpfloat_bias(self.bias_i)
    bias_h_q, adpbias_b_h = self.tl.get_adpfloat_bias(self.bias_h)

    # perform weight tiling
    wgt_i_qt = self.tl.lstm_wgt_tiling(wgt_i_q, self.num_v_in, self.num_v_out)
    wgt_h_qt = self.tl.lstm_wgt_tiling(wgt_h_q, self.num_v_out, self.num_v_out)
    print('*** performed weight matrix tiling for PEs ***')
    # perform bias reshape
    bias_i_qr = self.tl.lstm_bias_reshape(bias_i_q, self.num_v_out)
    bias_h_qr = self.tl.lstm_bias_reshape(bias_h_q, self.num_v_out)
    print('*** performed bias reshape for PEs ***')
    
    # these adpbias are very likely to be the same
    assert adpbias_wgt_i == adpbias_wgt_h, \
      'adpbias_wgt_i({}) != adpbias_wgt_h({})'.format(adpbias_wgt_i, adpbias_wgt_h)
    assert adpbias_b_i == adpbias_b_h, \
      'adpbias_b_i({}) != adpbias_b_h({})'.format(adpbias_b_i, adpbias_b_h)
    self.bias_wgt = int(adpbias_wgt_i + 10)
    self.bias_inp = int(adpbias_inp + 10)
    self.bias_b = int(adpbias_b_i + 10)
    self.bias_act = 2 # this is an empirical value
    print((self.bias_wgt, self.bias_inp, self.bias_b, self.bias_act))
    """
    init data_lib param
    """
    param = {
      'gb_num_vector_in' : self.num_v_in,
      'gb_num_vector_out' : self.num_v_out,
      'adpbias_wgt' : self.bias_wgt,
      'adpbias_inp' : self.bias_inp,
      'adpbias_bias' : self.bias_b,
      'adpbias_pe_act' : self.bias_act,
      'wi_num_tile' : int(4 * self.num_v_in * self.num_v_out),
      'wh_num_tile' : int(4 * self.num_v_out * self.num_v_out)
    }

    self.data_lib_to_adpfloat(wgt_i_qt, wgt_h_qt, inp_q, bias_i_qr, bias_h_qr)
    self.gen_data_lib_helper(param, './test/lstm_data_lib.json')


  def data_lib_to_adpfloat(self, wgt_i, wgt_h, inp, bias_i, bias_h):
    print('\n--------------------------------------------------------------')
    print('\tinvoking float to adpfloat converter')
    print('--------------------------------------------------------------\n')
    
    wgt_i.tofile('./test/wgt_i_qt.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/wgt_i_qt.tmp', self.bias_wgt, './test/wgt_i_qt_av.tmp')
    wgt_h.tofile('./test/wgt_h_qt.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/wgt_h_qt.tmp', self.bias_wgt, './test/wgt_h_qt_av.tmp')
    inp.tofile('./test/inp_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/inp_q.tmp', self.bias_inp, './test/inp_q_av.tmp')
    bias_i.tofile('./test/bias_i_qr.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/bias_i_qr.tmp', self.bias_b, './test/bias_i_qr_av.tmp')
    bias_h.tofile('./test/bias_h_qr.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/bias_h_qr.tmp', self.bias_b, './test/bias_h_qr_av.tmp')
    

  def gen_data_lib_helper(self, param, out_path):
    """
    produce data_lib for lstm
    """
    self.data_lib = param
    wi_num_tiles = param['wi_num_tile']
    wh_num_tiles = param['wh_num_tile']
    # TODO: dump weight, inp and bias data into data_lib
    with open('./test/wgt_i_qt_av.tmp', 'r') as fin:
      wgt_i_v_list = fin.read().splitlines()
    assert len(wgt_i_v_list) == wi_num_tiles * 16, \
      'list length {} != {}'.format(len(wgt_i_v_list), wi_num_tiles*16)
    self.data_lib = \
      self.tl.wgt_to_data_lib(wgt_i_v_list, 'wi', wi_num_tiles, self.data_lib)
    
    with open('./test/wgt_h_qt_av.tmp', 'r') as fin:
      wgt_h_v_list = fin.read().splitlines()
    assert len(wgt_h_v_list) == wh_num_tiles * 16, \
      'list length {} != {}'.format(len(wgt_h_v_list), wh_num_tiles*16)
    self.data_lib = \
      self.tl.wgt_to_data_lib(wgt_h_v_list, 'wh', wh_num_tiles, self.data_lib)
    
    with open('./test/inp_q_av.tmp', 'r') as fin:
      inp_v_list = fin.read().splitlines()
    assert len(inp_v_list) == self.num_v_in * self.num_ts
    for t in range(self.num_ts):
      self.data_lib = \
        self.tl.vector_to_data_lib(inp_v_list[t*self.num_v_in : (t+1)*self.num_v_in], \
                                   'ts_{}'.format(t), self.num_v_in, self.data_lib)
    
    with open('./test/bias_i_qr_av.tmp', 'r') as fin:
      bias_i_v_list = fin.read().splitlines()
    assert len(bias_i_v_list) == 4 * self.num_v_out, \
      'list length {} != {}'.format(len(bias_i_v_list), 4*self.num_v_out)
    self.data_lib = \
      self.tl.vector_to_data_lib(bias_i_v_list, 'bi', 4*self.num_v_out, self.data_lib)

    with open('./test/bias_h_qr_av.tmp', 'r') as fin:
      bias_h_v_list = fin.read().splitlines()
    assert len(bias_h_v_list) == 4 * self.num_v_out
    self.data_lib = \
      self.tl.vector_to_data_lib(bias_h_v_list, 'bh', 4*self.num_v_out, self.data_lib)
    
    with open(out_path, 'w') as fout:
      json.dump(self.data_lib, fout, indent=4)
    print('\n*** data_lib has been dump to {}***\n'.format(out_path))

  """
  produce reference LSTM data
  """
  def get_lstm_cell_data(self, i2h_wgt, i2h_bias, h2h_wgt, h2h_bias):
    self.W_ii = i2h_wgt[0 : 16*self.num_v_out, ]
    self.W_if = i2h_wgt[16*self.num_v_out : 2*16*self.num_v_out, ]
    self.W_ig = i2h_wgt[2*16*self.num_v_out : 3*16*self.num_v_out, ]
    self.W_io = i2h_wgt[3*16*self.num_v_out : 4*16*self.num_v_out, ]

    self.W_hi = h2h_wgt[0 : 16*self.num_v_out, ]
    self.W_hf = h2h_wgt[16*self.num_v_out : 2*16*self.num_v_out, ]
    self.W_hg = h2h_wgt[2*16*self.num_v_out : 3*16*self.num_v_out, ]
    self.W_ho = h2h_wgt[3*16*self.num_v_out : 4*16*self.num_v_out, ]

    self.b_ii = i2h_bias[0 : 16*self.num_v_out, ]
    self.b_if = i2h_bias[16*self.num_v_out : 2*16*self.num_v_out, ]
    self.b_ig = i2h_bias[2*16*self.num_v_out : 3*16*self.num_v_out, ]
    self.b_io = i2h_bias[3*16*self.num_v_out : 4*16*self.num_v_out, ]

    self.b_hi = h2h_bias[0 : 16*self.num_v_out, ]
    self.b_hf = h2h_bias[16*self.num_v_out : 2*16*self.num_v_out, ]
    self.b_hg = h2h_bias[2*16*self.num_v_out : 3*16*self.num_v_out, ]
    self.b_ho = h2h_bias[3*16*self.num_v_out : 4*16*self.num_v_out, ]

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def lstm_cell(self, input_state, prev_state, cell_state):
    i_t = self.sigmoid(np.add(np.add(np.matmul(self.W_ii, input_state), self.b_ii), \
                              np.add(np.matmul(self.W_hi, prev_state), self.b_hi)))
    f_t = self.sigmoid(np.add(np.add(np.matmul(self.W_if, input_state), self.b_if), \
                              np.add(np.matmul(self.W_hf, prev_state), self.b_hf)))
    g_t = np.tanh(np.add(np.add(np.matmul(self.W_ig, input_state), self.b_ig), \
                         np.add(np.matmul(self.W_hg, prev_state), self.b_hg)))
    o_t = self.sigmoid(np.add(np.add(np.matmul(self.W_io, input_state), self.b_io), \
                              np.add(np.matmul(self.W_ho, input_state), self.b_ho)))
    cell_state = np.add(np.dot(f_t, prev_state), np.dot(i_t, g_t))
    h_t = np.dot(o_t, np.tanh(cell_state))

    return h_t, cell_state
  

  def produce_ref_result(self):
    print('\n--------------------------------------------------------------')
    print('\tproducing reference LSTM results')
    print('--------------------------------------------------------------\n')
    # initial cell state
    cell_state = self.cell_state_init
    hidden_state = self.hidden_state_init
    self.get_lstm_cell_data(self.wgt_i, self.bias_i, self.wgt_h, self.bias_h)
    self.ref_out = []
    for t in range(self.num_ts):
      input = self.inp[t*16*self.num_v_in : (t+1)*16*self.num_v_in, ]
      hidden_state, cell_state = self.lstm_cell(input, hidden_state, cell_state)
      self.ref_out.append(hidden_state)

  # -----------------------------------------
  # invode ILA simulation
  # -----------------------------------------
  def gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr('./test/lstm_asm.json', './test/lstm_data_lib.json')
    self.ila_cvtr.dump_ila_prog_frag('./test/lstm_prog_frag_in.json')
    print('*** ILA program fragment has been dumped to ./test/lstm_prog_frag_in.json***\n')
  
  def invoke_ila_simulator(self):
    print('\n--------------------------------------------------------------')
    print('\tinvoking ILA simulator')
    print('--------------------------------------------------------------\n')
    subprocess.run(['./tool/asm_sim_driver.out',
                    './test/lstm_prog_frag_in.json',
                    './test/lstm_adpf_result.tmp'])

  def get_ila_sim_result(self):
    print('\n--------------------------------------------------------------')
    print('\tcollecting ILA simulation result')
    print('--------------------------------------------------------------\n')
    self.tl.axi_out_to_float('./test/lstm_adpf_result.tmp',
                             './test/lstm_float_result.tmp',
                             1, self.num_ts, self.num_v_in, self.num_v_out, self.bias_act)
  
    self.result = np.fromfile('./test/lstm_float_result.tmp', sep = '\n')
  
  # --------------------------------------
  # result analysis
  # --------------------------------------
  def result_analysis(self, is_verbose):
    print('\n--------------------------------------------------------------')
    print('\tanalyze ILA simulation result')
    print('--------------------------------------------------------------\n')
    for i in range(self.num_ts):
      result_ts = self.result[self.num_v_out*16*i : self.num_v_out*16*(i+1)]
      ref = self.ref_out[i]
      
      if is_verbose:
        print("reference output: \n{}\nresult: \n{}\n".format(ref, result_ts))

  # --------------------------------------
  # dump axi commands
  # --------------------------------------
  def gen_axi_cmds(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate axi commands for FlexNLP')
    print('--------------------------------------------------------------\n')
    if not self.ila_cvtr:
      self.ila_cvtr = cvtr('./test/lstm_asm.json', './test/lstm_data_lib.json')
    self.ila_cvtr.dump_axi_cmds('./test/lstm_axi_cmd.csv')
    print('*** axi commands has been dumped to ./test/lstm_axi_cmd.csv ***')