import json
import re
import sys
import numpy as np
import subprocess
import os
import argparse
from typing_extensions import OrderedDict

from base_driver import FlexASRBaseDriver
from src.converter import Converter as cvtr
from src.utils import tool

class linear_layer_driver(FlexASRBaseDriver):
  """
  If adptbias is set to None, then it would use the default auto-generated adaptive-float bias
  else, the adptbias is hard-coded to the value set below
  """
  # ADPTBIAS = -10
  ADPTBIAS = None

  def __init__(self, num_v_in, num_v_out, num_timestep, is_bias, dtype, op_name="", ref_run=False):
    super().__init__()
    self.num_v_in = num_v_in
    self.num_v_out = num_v_out
    self.num_ts = num_timestep
    self.is_bias = is_bias
    self.op_name = op_name
    self.dtype = dtype
    self.tl = tool()

    # add a ref_run flag for debugging
    self.ref_run = ref_run


  def produce_ly_asm(self, write_only=False):
    ila_asm = []
    ila_asm.append({
      'name' : 'store_wgt',
      'wgt_idx' : 'w0',
    })
    ila_asm.append({
      'name' : 'store_bias',
      'bias_idx' : 'b0',
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

    if not write_only:
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
    coef = 0.5
    wgt_init = coef*np.random.uniform(-1, 1, (16*self.num_v_out, 16*self.num_v_in)).astype(np.float32)
    inp_init = coef*np.random.uniform(-1, 1, (self.num_v_in * 16 * self.num_ts)).astype(np.float32)
    print('(wgt, inp) shape is ({},{})'.format(wgt_init.shape, tuple(t/self.num_ts for t in inp_init.shape)))

    if self.is_bias == 1:
      bias_init = coef*np.random.uniform(0, 1, (self.num_v_out*16)).astype(np.float32)
      print("\n**linear_layer bias is enabled**\n")
    else:
      bias_init = np.zeros((self.num_v_out*16, ), dtype=np.float32)
      print("\n**linear_layer bias is disabled, the following bias is a zero vector**\n")
    
    self.wgt = wgt_init
    self.inp = inp_init
    self.bias = bias_init

    # dump the test data to files 
    wgt_init.tofile("./data/wgt.txt", sep="\n")
    inp_init.tofile("./data/inp.txt", sep="\n")
    bias_init.tofile("./data/bias.txt", sep="\n")
  
  def collect_data_inp(self):
    # collecting input data TODO: replace the file path with correct one
    inp_path = './data/inp.txt'
    print('collecting input from ' + inp_path)
    self.inp = np.fromfile(inp_path, sep = '\n').astype(self.dtype)
  
  def collect_data_wgt(self):
    # TODO: do we need to transpose the wgt matrix here?
    wgt_path = './data/wgt.txt'
    print('collecting wgt from ' + wgt_path)
    wgt = np.fromfile(wgt_path, sep = '\n').astype(self.dtype)
    self.wgt = wgt.reshape((16*self.num_v_out, 16*self.num_v_in))
  
  def collect_data_bias(self):
    if self.is_bias == 1:
      bias_path = './data/bias.txt'
      print('collecting bias from ' + bias_path)
      self.bias = np.fromfile(bias_path, sep = '\n').astype(self.dtype)
    else:
      self.bias = np.zeros(self.num_v_out*16).astype(self.dtype)

  def collect_data(self):
    print('\n--------------------------------------------------------------')
    print('\tcollecting input data')
    print('--------------------------------------------------------------\n')    
    self.collect_data_inp()
    self.collect_data_wgt()
    self.collect_data_bias()

  def produce_ref_result(self):
    # -------------------------
    # get reference output
    # -------------------------
    if self.dtype == "int8":
      self.bias_act = 0
      return
    self.ref = []
    for i in range(self.num_ts):
      inp_ts = self.inp[self.num_v_in*16*i : self.num_v_in*16*(i+1)]
      ref = np.add(np.matmul(self.wgt, inp_ts), self.bias)
      ref_q, bias_act = self.tl.get_adpfloat_bias(ref, self.ADPTFLOAT_N_BITS, self.ADPTFLOAT_N_EXP, self.ADPTBIAS)
      self.bias_act = bias_act
      # need to quantize the reference as well for comparsion
      # self.ref.append(ref_q)
      self.ref.append(ref)


  def produce_ly_data_lib(self):
    # if the dtype is float32, then use the normal adaptive-float numerics from FlexASR design
    if self.dtype == "float32":
      print("\n*** Taking float32 inputs! ***\n")
      # --------------------------
      # get quantized inputs & weight tiling
      # --------------------------
      wgt_q, bias_wgt = self.tl.get_adpfloat_bias(self.wgt, self.ADPTFLOAT_N_BITS, self.ADPTFLOAT_N_EXP, self.ADPTBIAS)
      inp_q, bias_inp = self.tl.get_adpfloat_bias(self.inp, self.ADPTFLOAT_N_BITS, self.ADPTFLOAT_N_EXP, self.ADPTBIAS)
      bias_q, bias_b = self.tl.get_adpfloat_bias(self.bias, self.ADPTFLOAT_N_BITS, self.ADPTFLOAT_N_EXP, self.ADPTBIAS)

      print('\n*** performed weight matrix tiliing for PEs***\n')
      wgt_qt = self.tl.wgt_tiling(wgt_q, self.num_v_in, self.num_v_out)

      self.bias_wgt = int(bias_wgt + self.ADPTFLOAT_OFFSET)
      self.bias_inp = int(bias_inp + self.ADPTFLOAT_OFFSET)
      self.bias_b = int(bias_b + self.ADPTFLOAT_OFFSET)
      self.bias_act = int(self.bias_act + self.ADPTFLOAT_OFFSET)
      print(f"{self.bias_wgt}::{self.bias_b}::{self.bias_inp}::{self.bias_act}")

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
        'w0_num_tile' : int(self.num_v_in * self.num_v_out),
      }

      self.wgt = wgt_q
      self.inp = inp_q
      self.bias = bias_q

      print('\n--------------------------------------------------------------')
      print('\tinvoking float to adpfloat converter')
      print('--------------------------------------------------------------\n')
      
      wgt_qt.tofile('./test/wgt_qt.tmp', sep = '\n')
      self.tl.call_float_adpt_v_cvtr('./test/wgt_qt.tmp', self.bias_wgt, './test/wgt_qt_av.tmp')
      inp_q.tofile('./test/inp_q.tmp', sep = '\n')
      self.tl.call_float_adpt_v_cvtr('./test/inp_q.tmp', self.bias_inp, './test/inp_q_av.tmp')
      bias_q.tofile('./test/bias_q.tmp', sep = '\n')
      self.tl.call_float_adpt_v_cvtr('./test/bias_q.tmp', self.bias_b, './test/bias_q_av.tmp')
  
    # if the data_type is int8, then no need to convert the data to adaptive-float
    elif self.dtype == "int8":
      print("\n*** Taking int8 inputs! ***")
      print("\n*** performing weight matrix tiling for PEs***\n")
      self.wgt = self.tl.wgt_tiling(self.wgt, self.num_v_in, self.num_v_out)
      # ---------------------------
      # init data_lib param
      # ---------------------------
      param = {
        'gb_num_vector_in' : self.num_v_in,
        'gb_num_vector_out' : self.num_v_out,
        'adpbias_wgt' : 0,
        'adpbias_inp' : 0,
        'adpbias_bias' : 0,
        'adpbias_pe_act' : 0,
        'w0_num_tile' : int(self.num_v_in * self.num_v_out),
      }
      
      # print data to file, borrow the same name as the float32 version for convenience
      self.wgt.tofile("./test/wgt.tmp", sep="\n")
      self.tl.call_pack_int8_to_vector("./test/wgt.tmp", "./test/wgt_qt_av.tmp")
      self.inp.tofile("./test/inp.tmp", sep="\n")
      self.tl.call_pack_int8_to_vector("./test/inp.tmp", "./test/inp_q_av.tmp")
      self.bias.tofile("./test/bias.tmp", sep="\n")
      self.tl.call_pack_int8_to_vector("./test/bias.tmp", "./test/bias_q_av.tmp")

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
      self.tl.wgt_to_data_lib(wgt_v_list, 'w0', self.num_v_in*self.num_v_out, self.data_lib)

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
    
    
  def produce_linear_layer_test(self, write_only=False):
    self.produce_random_test_data()
    self.collect_data()
    self.produce_ref_result()
    self.produce_ly_data_lib()
    self.produce_ly_asm(write_only)
  
  # -----------------------------------------
  # invoke ILA simulation
  # -----------------------------------------
  def gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr('./test/ly_asm.json', './test/ly_data_lib.json')
    self.ila_cvtr.dump_ila_asm('./test/ly_ila_asm.json')
    self.ila_cvtr.dump_ila_prog_frag('./test/ly_prog_frag_in.json')
    print('***ILA program fragment has been dumped to ./test/ly_prog_frag_in.json***\n')
  

  def collect_ila_result(self):
    """
    run ila simulation and collect the result
    """
    self.result_ila = self.tl.collect_ila_result(
      in_path='./test/ly_prog_frag_in.json',
      mem_idx=1, 
      num_ts=self.num_ts, 
      num_vi=self.num_v_in, 
      num_vo=self.num_v_out, 
      bias=self.bias_act,
      dtype=self.dtype,
    )


  def result_analysis(self, is_verbose = 0, is_fpga = 0):
    print('\n--------------------------------------------------------------')
    print('\tanalyze ILA simulation result')
    print('--------------------------------------------------------------\n')
    err_ref_list = []
    ts_stdd_list = []
    for i in range(self.num_ts):
      if is_fpga:
        result_ts = self.result_fpga[self.num_v_out*16*i : self.num_v_out*16*(i+1)]
      else:
        result_ts = self.result_ila[self.num_v_out*16*i : self.num_v_out*16*(i+1)]
      ref = self.ref[i]
      avg_mm = self.tl.cal_error_single_tensor(result_ts, ref)
      print(f"result timestep No.{i} --- relative error (vs. ref): {avg_mm:.5%}")
      if is_verbose:
        print("reference output: \n{}\nresult: \n{}\n".format(ref, result_ts))
      err_ref_list.append(avg_mm)
      ts_stdd_list.append(0)

    return err_ref_list, ts_stdd_list

  # --------------------------------------
  # invoke FPGA simulation
  # --------------------------------------
  def gen_axi_cmds(self, base_addr = '0x33000000', fname = './test/ly_axi_cmd.csv'):
    """
    dump FPGA axi commands and simulatino c script
    the default address here is 0x33000000 is for running flexnlp systemc simulation
    """
    print('\n--------------------------------------------------------------')
    print('\tgenerate axi commands for FlexNLP')
    print('--------------------------------------------------------------\n')
    if not self.ila_cvtr:
      self.ila_cvtr = cvtr('./test/ly_asm.json', './test/ly_data_lib.json')
    self.ila_cvtr.dump_axi_cmds(fname, base_addr, op_name=self.op_name)
    print(f'*** axi commands has been dumped to {fname} ***')


  def collect_fpga_results(self, base_addr = '0xA0500000'):
    """
    run FlexNLP FPGA simulation and collect the results
    TODO: base address here should set as 0xA0500000, which is the base address of the 
    GB large buffer of FlexNLP on FPGA
    """
    self.result_fpga = self.tl.collect_fpga_results(
      mem_idx=1, 
      num_ts=self.num_ts,
      num_vi=self.num_v_in, 
      num_vo=self.num_v_out, 
      bias=self.bias_act,
      base_addr=base_addr, 
      op_name=self.op_name
    )
    

  def run(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self.collect_data()
    if not self.ref_run:
      # driver needs producing reference result is to get output activation adpbias
      self.produce_ref_result()
      self.produce_ly_data_lib()
      self.produce_ly_asm()
      self.gen_prog_frag()
      if not os.getenv('USE_3LA_FPGA') in ('1', 'ON'):
        self.collect_ila_result()
        self.result_ila.tofile('./data/result.txt', sep='\n')
      else:
        self.gen_axi_cmds('0xA0000000')
        self.collect_fpga_results()
        self.result_fpga.tofile('./data/result.txt', sep = '\n')
    
    else:
      # run a software reference run and return the results
      self.produce_ref_result()
      np.asarray_chkfinite(self.ref).tofile("./data/result.txt", sep="\n")
    
    if os.getenv("TVM_3LA_DIFF_DEBUG"):
      if self.ref_run:
        self.collect_imm_result("3la_layer_ref_imm_results.json")
      else:
        self.collect_imm_result("3la_layer_imm_results.json")

  
  def collect_imm_result(self, fname):
    """
    collect intermediate results for per layer analysis
    """
    if not os.path.exists(fname):
      json.dump({}, open(fname, "w"))
    tensor_in = np.fromfile("./data/inp.txt", sep='\n')
    tensor_out = np.fromfile("./data/result.txt", sep='\n')
    imm_result_tb = json.load(open(fname, "r"))
    imm_result_tb[self.op_name] = {
      # "in" : tensor_in.tolist(),
      "in" : self.inp.tolist(),
      "out" : tensor_out.tolist(),
      "wgt_range" : f"{np.min(self.wgt)}, {np.max(self.wgt)}",
      "bias_range" : f"{np.min(self.bias)}, {np.max(self.bias)}",
    }
    with open(fname, "w") as fo:
      json.dump(imm_result_tb, fo)

  

  def run_test(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self.produce_linear_layer_test()
    self.gen_prog_frag()
    self.collect_ila_result()
    self.gen_axi_cmds('0xA0000000')
    self.result_ila.tofile('./data/result_ila_sim.txt', sep='\n')
    return self.result_analysis()
    # err_ref_list = self.result_analysis()
    # return err_ref_list
    # self.collect_fpga_results()
    # self.result_fpga.tofile('./data/result_fpga_sim.txt', sep = '\n')

  def clean_up(self):
    for file in os.listdir('./test'):
      if '.tmp' in file:
        subprocess.run(['rm', './test/'+file])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="FlexASR Linear Layer Driver")
  parser.add_argument("--num_v_in", type=int, required=True, 
                      help="number of vector in the input timesteps")
  parser.add_argument("--num_v_out", type=int, required=True,
                      help="number of vector in the output timesteps")
  parser.add_argument("--num_timestep", type=int, required=True, 
                      help="number of timesteps of the linear layer")
  parser.add_argument("--is_bias", type=bool, required=True,
                      help="whether to apply bias to the linear layer")
  parser.add_argument("--dtype", type=str, required=True, choices=["float32", "int8"],
                      help="Specify data type of the computation")
  parser.add_argument("--op_name", type=str, default="linear_layer")
  # add an argument for return a sw reference result
  parser.add_argument("--ref_run", type=bool, default=False)

  kwargs = vars(parser.parse_args())

  driver = linear_layer_driver(**kwargs)
  driver.run()
  driver.clean_up()  