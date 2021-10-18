import json
import sys
import numpy as np
import subprocess
import os

from src.utils import tool
from src.converter import Converter as cvtr

class pooling_layer_driver:
  def __init__(self, mode, num_v_in, num_ts):
    """
    for pooling layers, num_v_in == num_v_out,
    mode: 'max': maxpooling; 'mean': mean-pooling
    """
    assert mode == 'max' or mode == 'mean', \
      'Unsupported layer-pooling operator'
    self.num_v_in = num_v_in
    self.num_ts = num_ts
    self.mode = mode
    self.tl = tool()
  
  def produce_pooling_asm(self):
    ila_asm = []
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'store_act',
        'timestep_idx' : 'ts_' + str(i),
        'idx' : i
      })
    if self.mode == 'max':
      ila_asm.append({
        'name' : 'maxp',
        'num_ts' : self.num_ts
      })
    elif self.mode == 'mean':
      ila_asm.append({
        'name' : 'meanp',
        'num_ts' : self.num_ts
      })
    for i in range(self.num_ts >> 1):
      ila_asm.append({
        'name' : 'load_act',
        'mem_idx' : 0,
        'ts_idx' : i
      })
    
    ila_asm = {'asm' : ila_asm}
    self.ts_asm_lib = ila_asm

    with open('./test/pooling_asm.json', 'w') as fout:
      json.dump(ila_asm, fout, indent=4)
    print('*** ILA tensor assembly has been dumped to ./test/pooling_asm.json ***')
  
  def produce_pooling_data_lib(self):
    """
    get quantized inputs
    """
    inp_q, adpbias_inp = self.tl.get_adpfloat_bias(self.inp)
    self.bias_inp = int(adpbias_inp + 10)
    param = {
      'gb_num_vector_in' : self.num_v_in,
      'gb_num_vector_out' : self.num_v_in,
      'adpbias_inp' : self.bias_inp
    }

    print('\n--------------------------------------------------------------')
    print('\tinvoking float to adpfloat converter')
    print('--------------------------------------------------------------\n')

    inp_q.tofile('./test/inp_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/inp_q.tmp', self.bias_inp, './test/inp_q_av.tmp')

    self.gen_data_lib_helper(param, './test/pooling_data_lib.json')

    """
    change the original data to quantized data, for better result matching
    """
    self.inp = inp_q

  def gen_data_lib_helper(self, param, out_path):
    """
    produce data_lib for pooling
    """
    self.data_lib = param
    with open('./test/inp_q_av.tmp') as fin:
      inp_v_list = fin.read().splitlines()
    assert len(inp_v_list) == self.num_v_in * self.num_ts
    for t in range(self.num_ts):
      self.data_lib = \
        self.tl.vector_to_data_lib(inp_v_list[t*self.num_v_in : (t+1)*self.num_v_in], 
                                   'ts_{}'.format(t), self.num_v_in, self.data_lib)
    with open(out_path, 'w') as fout:
      json.dump(self.data_lib, fout, indent=4)
    print('\n*** data_lib has been dump to {}***\n'.format(out_path))


  # -----------------------------------------
  # invoke ILA simulation
  # -----------------------------------------
  def gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr('./test/pooling_asm.json', './test/pooling_data_lib.json')
    self.ila_cvtr.dump_ila_asm('./test/pooling_ila_asm.json')
    self.ila_cvtr.dump_ila_prog_frag('./test/pooling_prog_frag_in.json')
    print('*** ILA program fragment has been dumped to ./test/pooling_prog_frag_in.json***\n')

  def collect_ila_result(self):
    """
    run ila simulation and collect the result
    """
    self.result_ila = self.tl.collect_ila_result(in_path='./test/pooling_prog_frag_in.json',
                      mem_idx=0, num_ts=self.num_ts >> 1, 
                      num_vi=self.num_v_in, num_vo=self.num_v_in, bias=self.bias_inp)

  # --------------------------------------
  # dump axi commands
  # --------------------------------------
  def gen_axi_cmds(self, base_addr):
    print('\n--------------------------------------------------------------')
    print('\tgenerate axi commands for FlexNLP')
    print('--------------------------------------------------------------\n')
    if not self.ila_cvtr:
      self.ila_cvtr = cvtr('./test/pooling_asm.json', './test/pooling_data_lib.json')
    self.ila_cvtr.dump_axi_cmds('./test/pooling_axi_cmd.csv', base_addr)
    print('*** axi commands has been dumped to ./test/pooling_axi_cmd.csv ***')

  def run_test(self, verbose_analysis=0):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self.produce_pooling_asm()
    self.produce_random_test_data()
    self.produce_pooling_data_lib()
    self.gen_prog_frag()
    self.collect_ila_result()
    self.gen_axi_cmds('0xA0000000')
    self.produce_ref_result()
    return self.result_analysis()
    # err_ref_list = self.result_analysis(verbose_analysis)
    # return err_ref_list
  

  ##########################################
  ##  Test tool
  ##########################################

  # ----------------------------------------
  # produce pooling test data
  # ----------------------------------------
  def produce_random_test_data(self):
    """
    produce random data for testing
    """
    print('\n--------------------------------------------------------------')
    print('\tproducing random input data')
    print('--------------------------------------------------------------\n')
    coef = 1

    self.inp = \
      coef * np.random.uniform(-1, 1, (self.num_ts * 16 * self.num_v_in)).astype(np.float32)
  
  def produce_ref_result(self):
    """
    reshape the input data to each timestep
    """
    inp = self.inp.reshape((self.num_ts, 16*self.num_v_in))
    out = []
    for i in range(self.num_ts >> 1):
      if self.mode == 'max':
        out.append(np.maximum(inp[2*i, ], inp[2*i+1, ]))
      elif self.mode == 'mean':
        out.append((inp[2*i, ] + inp[2*i+1, ])/2)
    self.ref_out = out
  
  def result_analysis(self, is_verbose=0):
    print('\n--------------------------------------------------------------')
    print('\tanalyze ILA simulation result')
    print('--------------------------------------------------------------\n')
    err_ref_list = []
    ts_stdd_list = []
    for i in range(self.num_ts >> 1):
      if not os.environ.get('USE_3LA_FPGA'):
        result_ts = self.result_ila[self.num_v_in*16*i : self.num_v_in*16*(i+1)]
      ref = self.tl.get_adpfloat_bias(self.ref_out[i])[0]
      err_out, err_ref = self.tl.cal_error(result_ts, ref)
      print("result timestep No.{} --- relative error (vs. sim_out): {:5.5%}\
            relative error (vs. ref): {:5.5%}\n".format(i, err_out, err_ref))
      if is_verbose:
        print("reference output: \n{}\nresult: \n{}\n".format(ref, result_ts))
      # err_ref_list.append(err_ref)

      avg_mm, ts_stdd = self.tl.cal_error_single_tensor(result_ts, ref)
      err_ref_list.append(avg_mm)
      ts_stdd_list.append(ts_stdd)


      # if self.num_ts == 2:
      #   avg_mm, stdd = self.tl.cal_error_single_tensor(result_ts, ref)
      #   print(f"Summary of a single tensor: Mean: {avg_mm:.5%}\t Standard deviation: {stdd:.5%}")
    
    # mean, stdd = self.tl.cal_mean_stdd(err_ref_list)
    # print(f"Summary of Mismatch: Mean: {mean:.5%}\t Standard Deviation: {stdd:.5%}.")
    return err_ref_list, ts_stdd_list

  def clean_up(self):
    for file in os.listdir('./test'):
      if '.tmp' in file:
        subprocess.run(['rm', './test/'+file])


if __name__ == '__main__':
  assert len(sys.argv) == 4, \
    "Usage: python3 pooling_driver.py [mode] [num_vector_in] [num_timestep]"
  mode = sys.argv[1]
  num_v_in = int(sys.argv[2])
  num_ts = int(sys.argv[3])

  test_driver = pooling_layer_driver(mode, num_v_in, num_ts)
  # test_driver.run()
  test_driver.run_test(verbose_analysis=1)
  test_driver.clean_up()