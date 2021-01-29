import json
import sys
import numpy as np
import subprocess
import os

from utils import tool as tool
from converter import Converter as cvtr

class linear_layer_driver:
  def __init__(self, num_v_in, num_v_out, num_ts, is_bias):
    self.num_v_in = num_v_in
    self.num_v_out = num_v_out
    self.num_ts = num_ts
    self.is_bias = is_bias
    self.tl = tool()

  def produce_ly_asm(self):
    ila_asm = []
    ila_asm.append({
      'name' : 'store_wgt',
      'wgt_idx' : 'w0'
    })
    ila_asm.append({
      'name' : 'store_bias',
      'bias_idx' : 'b0'
    })
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'store_act',
        'timestep_idx' : 'ts_' + str(i),
        'idx' : i
      })
    ila_asm.append({
      'name' : 'linear_layer',
      'num_ts' : self.num_ts,
      'is_bias' : self.is_bias
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

    with open('./test/ly_asm.json', 'w') as fout:
      json.dump(ila_asm, fout, indent=4)
    print('*** ILA tensor assembly has been dumped to ./test/ly_asm.json ***\n')

  def produce_random_test_data(self):
    """
    produce random data for testing
    """
    print('\n--------------------------------------------------------------')
    print('\tproducing random input data')
    print('--------------------------------------------------------------\n')
    coef = 0.2
    wgt_init = coef*np.random.random_sample((16*self.num_v_out, 16*self.num_v_in))
    inp_init = coef*np.random.random_sample((self.num_v_in * 16 * self.num_ts))
    print('(wgt, inp) shape is ({},{})'.format(wgt_init.shape, tuple(t/self.num_ts for t in inp_init.shape)))

    if self.is_bias == 1:
      bias_init = coef*np.random.random_sample((self.num_v_out*16))
      print("\n**linear_layer bias is enabled**\n")
    else:
      bias_init = np.zeros(self.num_v_out*16)
      print("\n**linear_layer bias is disabled, the following bias is a zero vector**\n")
    
    self.wgt = wgt_init
    self.inp = inp_init
    self.bias = bias_init
  
  def collect_data(self):
    print('\n--------------------------------------------------------------')
    print('\tcollecting input data')
    print('--------------------------------------------------------------\n')
    
    # collecting input data TODO: replace the file path with correct one
    inp_path = './data/inp.txt'
    print('collecting input from ' + inp_path)
    self.inp = np.fromfile(inp_path, sep = '\n')

    # TODO: do we need to transpose the wgt matrix here?
    wgt_path = './data/wgt.txt'
    print('collecting wgt from ' + wgt_path)
    wgt = np.fromfile(wgt_path, sep = '\n')
    self.wgt = wgt.reshape((16*self.num_v_out, 16*self.num_v_in))

    if self.is_bias == 1:
      bias_path = './data/bias.txt'
      print('collecting bias from ' + bias_path)
      self.bias = np.fromfile(bias_path, sep = '\n')
    else:
      self.bias = np.zeros(self.num_v_out*16)

  def produce_ref_result(self):
    # -------------------------
    # get reference output
    # -------------------------
    for i in range(self.num_ts):
      inp_ts = self.inp[self.num_v_in*16*i : self.num_v_in*16*(i+1)]
      # print(inp_ts.shape)
      # print(self.wgt.shape)

      ref = np.add(np.matmul(self.wgt, inp_ts), self.bias)
      ref.tofile('./test/ref_{}.tmp'.format(i), sep = '\n')
      # print("reference output No.{}".format(i))
      # print(ref)
      ref_q, bias_act = self.tl.get_adpfloat_bias(ref)
      # print("bias_act is {}\n".format(bias_act + 10))
      self.bias_act = bias_act

  def produce_ly_data_lib(self):
    # --------------------------
    # get quantized inputs & weight tiling
    # --------------------------
    wgt_q, bias_wgt = self.tl.get_adpfloat_bias(self.wgt)
    inp_q, bias_inp = self.tl.get_adpfloat_bias(self.inp)
    bias_q, bias_b = self.tl.get_adpfloat_bias(self.bias)

    print('\n*** performed weight matrix tiliing for PEs***\n')
    wgt_qt = self.tl.wgt_tiling(wgt_q, self.num_v_in, self.num_v_out)

    self.bias_wgt = int(bias_wgt + 10)
    self.bias_inp = int(bias_inp + 10)
    self.bias_b = int(bias_b + 10)
    self.bias_act = int(self.bias_act + 10)
    # ---------------------------
    # init data_lib param
    # ---------------------------
    param = {
      'gb_num_vector_in' : self.num_v_in,
      'gb_num_vector_out' : self.num_v_out,
      'adpbias_wgt' : self.bias_wgt,
      'adpbias_inp' : self.bias_inp,
      'adpbias_bias' : self.bias_b,
      'adpbias_pe_act' : self.bias_act,
      'w0_num_tile' : int(self.num_v_in * self.num_v_out)
    }

    print('\n--------------------------------------------------------------')
    print('\tinvoking float to adpfloat converter')
    print('--------------------------------------------------------------\n')
    
    wgt_qt.tofile('./test/wgt_qt.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/wgt_qt.tmp', self.bias_wgt, './test/wgt_qt_av.tmp')
    inp_q.tofile('./test/inp_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/inp_q.tmp', self.bias_inp, './test/inp_q_av.tmp')
    bias_q.tofile('./test/bias_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/bias_q.tmp', self.bias_b, './test/bias_q_av.tmp')
    
    self.gen_data_lib_helper(param, './test/ly_data_lib.json')
  
  def gen_data_lib_helper(self, param, out_path):
    """
    produce data_lib for ly
    """
    self.data_lib = param
    
    with open('./test/wgt_qt_av.tmp', 'r') as fin:
      wgt_v_list = fin.read().splitlines()
    assert len(wgt_v_list) % 16 == 0
    self.data_lib = \
      self.tl.wgt_to_data_lib(wgt_v_list, 'w0', self.num_v_in, self.num_v_out, self.data_lib)

    with open('./test/inp_q_av.tmp', 'r') as fin:
      inp_v_list = fin.read().splitlines()
    assert len(inp_v_list) == self.num_v_in * self.num_ts
    for t in range(self.num_ts):
      self.data_lib = \
        self.tl.vector_to_data_lib(inp_v_list[t * self.num_v_in : (t + 1) * self.num_v_in], \
                                   'ts_{}'.format(t), self.num_v_in, self.data_lib)
    
    with open('./test/bias_q_av.tmp', 'r') as fin:
      bias_v_list = fin.read().splitlines()
    assert len(bias_v_list) == self.num_v_out
    self.data_lib = \
      self.tl.vector_to_data_lib(bias_v_list, 'b0', self.num_v_out, self.data_lib)

    with open(out_path, 'w') as fout:
      json.dump(self.data_lib, fout, indent=4)

    print('\n\t*** data_lib has been dump to {}***\n'.format(out_path))
    
    
  def produce_linear_layer_test(self):
    self.produce_random_test_data()
    self.produce_ref_result()
    self.produce_ly_data_lib()
    self.produce_ly_asm()
  
  # -----------------------------------------
  # invoke ILA simulation
  # -----------------------------------------
  def gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    ila_cvtr = cvtr('./test/ly_asm.json', './test/ly_data_lib.json')
    ila_cvtr.dump_ila_prog_frag('./test/ly_prog_frag_in.json')
    print('***ILA program fragment has been dumped to ./test/ly_prog_frag_in.json***\n')
  
  def invoke_ila_simulator(self):
    print('\n--------------------------------------------------------------')
    print('\tinvoking ILA simulator')
    print('--------------------------------------------------------------\n')
    subprocess.run(['./tool/asm_sim_driver.out',
                    './test/ly_prog_frag_in.json',
                    './test/ly_adpf_result.tmp'])
  
  def get_ila_sim_result(self):
    print('\n--------------------------------------------------------------')
    print('\tcollecting ILA simulation result')
    print('--------------------------------------------------------------\n')
    self.tl.axi_out_to_float('./test/ly_adpf_result.tmp',
                             './test/ly_float_result.tmp',
                             1, self.num_ts, self.num_v_in, self.num_v_out, self.bias_act)
  
    self.result = np.fromfile('./test/ly_float_result.tmp', sep = '\n')
  
  def result_analysis(self):
    print('\n--------------------------------------------------------------')
    print('\tanalyze ILA simulation result')
    print('--------------------------------------------------------------\n')
    for i in range(self.num_ts):
      result_ts = self.result[self.num_v_out*16*i : self.num_v_out*16*(i+1)]
      print(result_ts)
      ref = np.fromfile('./test/ref_{}.tmp'.format(i), sep = '\n')
      err_out, err_ref = self.tl.cal_error(result_ts, ref)
      print("result timestep No.{} --- relative error (vs. sim_out): {:5.5%}\
            relative error (vs. ref): {:5.5%}\n".format(i, err_out, err_ref))

  def run(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test'])
    self.collect_data()
    self.produce_ref_result()
    self.produce_ly_data_lib()
    self.produce_ly_asm()
    self.gen_prog_frag()
    self.invoke_ila_simulator()
    self.get_ila_sim_result()
    # self.result_analysis()
    # dump result
    self.result.tofile('./data/result.txt', sep = '\n')
  
  def run_test(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test'])
    self.produce_linear_layer_test()
    self.gen_prog_frag()
    self.invoke_ila_simulator()
    self.get_ila_sim_result()
    self.result_analysis()

  def clean_up(self):
    for file in os.listdir('./test'):
      if '.tmp' in file:
        subprocess.run(['rm', './test/'+file])


if __name__ == '__main__':
  assert len(sys.argv) == 5, \
    "Usage: python3 linear_layer_driver.py [num_vector_in] [num_vector_out] [num_timestep] [is_bias]"
  num_v_in = int(sys.argv[1])
  num_v_out = int(sys.argv[2])
  num_ts = int(sys.argv[3])
  is_bias = int(sys.argv[4])

  driver = linear_layer_driver(num_v_in, num_v_out, num_ts, is_bias)
  driver.run()
  # driver.result_analysis()
  driver.clean_up()