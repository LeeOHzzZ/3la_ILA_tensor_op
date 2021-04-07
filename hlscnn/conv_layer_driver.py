"""
python driver for hlscnn-ila simulator
"""

import json
import sys
import numpy as np
import subprocess
import os
import argparse
import timeit

from converter import Converter as cvtr

class conv_layer_driver:
  __VIR_MEM_BASE_ADDR = 0x50000
  __SPAD0_BASE_ADDR = 0x04000
  __SPAD1_BASE_ADDR = 0X24000

  def __init__(self, inp_size, out_size, kernel_size, stride,
               is_bias, bias, is_relu, is_accum):
    self.inp_chans, self.inp_rows, self.inp_cols = inp_size
    self.out_chans, self.out_rows, self.out_cols = out_size
    self.k_num, self.k_chan, self.kernel_rows, self.kernel_cols = kernel_size
    self.k_r_stride, self.k_c_stride = stride
    self.is_bias = is_bias
    self.bias = bias
    self.is_relu = is_relu
    self.is_accum = is_accum
    self.ila_asm = []
    self.data_lib = []
    assert self.inp_chans == self.k_chan, 'input channels not equal to kernel channels'
    assert self.k_num == self.out_chans, 'output channels not equal to kernel numbers'
  
  def produce_vir_mem_wr_asm(self):
    """
    this function produce vir_mem_wr asm for transferring weights and activations
    """
    for i, data in enumerate(self.act_mem):
      self.ila_asm.append({
        'name' : 'VirMemWr',
        'addr' : hex(self.__VIR_MEM_BASE_ADDR + i*0x10),
        'data' : data
      })

    for i, data in enumerate(self.wgt_mem):
      self.ila_asm.append({
        'name' : 'VirMemWr',
        'addr' : hex(self.__VIR_MEM_BASE_ADDR + self.conv_wgt_soc_offset + i*0x10),
        'data' : data
      })
  
  def produce_spad_wr_asm(self):
    """
    this function produce spad_wr asm for writing the weight data into spad
    """
    # Warning: this may not work for HLSCNN systemc simulation when RdWrLen is larger than 1
    # Warning: If to run hlscnn systemc simulation, we have to write to spad1 to initialize it to 0
    self.ila_asm.append({
      'name' : 'CfgSoCMemRdWrLen',
      'length' : len(self.wgt_mem)
    })
    # assume weight data starts from 0x0 in the SoC Memory (0x50000 for VirMem)
    self.ila_asm.append({
      'name' : 'CfgSoCMemBaseAddr',
      'base_addr' : hex(self.conv_wgt_soc_offset)
    })
    self.ila_asm.append({
      'name' : 'SpadWr',
      'addr' : hex(self.__SPAD0_BASE_ADDR)
    })

  def produce_conv_layer_asm(self):
    """
    this three base address of HLSCNN are set as fixed value for now
    """
    self.ila_asm.append({
      'name' : 'CfgConvActBaseAddr',
      'base_addr' : hex(0x0)
    })
    self.ila_asm.append({
      'name' : 'CfgConvWgtBaseAddr',
      'base_addr' : hex(self.__SPAD0_BASE_ADDR)
    })
    self.ila_asm.append({
      'name' : 'CfgConvOutBaseAddr',
      'base_addr' : hex(self.__SPAD1_BASE_ADDR)
    })
    """
    Conv layer parameters
    """
    self.ila_asm.append({
      'name' : 'CfgConvInpSize',
      'inp_cols' : self.inp_cols,
      'inp_rows' : self.inp_rows,
      'inp_chans' : self.inp_chans
    })
    self.ila_asm.append({
      'name' : 'CfgConvOutSize',
      'out_cols' : self.out_cols,
      'out_rows' : self.out_rows,
      'out_chans' : self.out_chans
    })
    self.ila_asm.append({
      'name' : 'CfgConvKernelSize',
      'kernel_cols' : self.kernel_cols,
      'kernel_rows' : self.kernel_rows,
      'kernel_c_stride' : self.k_c_stride,
      'kernel_r_stride' : self.k_r_stride
    })
    self.ila_asm.append({
      'name' : 'CfgConvChan',
      'chan_bias' : self.bias,
      'is_bias' : self.is_bias,
      'is_relu' : self.is_relu,
      'is_accum' : self.is_accum,
      'kernel_num' : self.k_num,
      'is_wb' : 0
    })
    self.ila_asm.append({
      'name' : 'ConvStart'
    })
  
  def produce_read_asm(self):
    """
    produce asm for reading data from HLSCNN
    """
    # reference from conv_test_standalone in host.h by Harvard
    # out_idx is the idx of the output element in the output matrix
    self.out_addr = [] # holding the address of the output element
    
    for n in range(self.k_num):
      for i in range(self.out_rows):
        for j in range(self.out_cols):
          in_row = int(i*self.k_r_stride + (self.kernel_rows-1)/2)
          in_col = int(j*self.k_c_stride + (self.kernel_cols-1)/2)
          spad_idx = n*self.inp_rows*self.inp_cols + in_row*self.inp_cols + in_col
          spad_addr = spad_idx * 0x10 + self.__SPAD1_BASE_ADDR
          # append the read instructions
          self.ila_asm.append({
            'name' : 'SpadRd',
            'addr' : hex(spad_addr)
          })
          self.out_addr.append(spad_addr + ((self.k_num-1)%8) * 2)
  
  def produce_asm_all(self):
    self.produce_vir_mem_wr_asm()
    self.produce_spad_wr_asm()
    self.produce_conv_layer_asm()
    self.produce_read_asm()

    self.ila_asm = {'asm' : self.ila_asm}
    
    with open('./test/conv_ila_asm.json', 'w') as fout:
      json.dump(self.ila_asm, fout, indent=2)
    with open('./test/conv_ila_data_lib.json', 'w') as fout:
      json.dump(self.data_lib, fout, indent=2)

  def collect_data_in(self):
    """
    collect relay data from files
    """
    print('\n--------------------------------------------------------------')
    print('\tcollecting input data')
    print('--------------------------------------------------------------\n')
    cmd = ['hlscnn_pack_data',
           str(self.inp_rows), str(self.inp_cols), str(self.inp_chans),
           str(self.kernel_rows), str(self.kernel_cols), str(self.k_num)]
    subprocess.run(cmd)

    with open('./data/packed_conv_act.json', 'r') as fin:
      self.act_mem = json.load(fin)
    with open('./data/packed_conv_wgt.json', 'r') as fin:
      self.wgt_mem = json.load(fin)

    # self.conv_act_offset = (len(self.wgt_mem) + 1) * 0x10
    self.conv_wgt_soc_offset = int(len(self.act_mem)/16 + 1) * 16 * 0x10

  def produce_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr('./test/conv_ila_asm.json', './test/conv_ila_data_lib.json')
    self.ila_cvtr.dump_ila_prog_frag('./test/conv_ila_prog_frag.json')
  
  def invoke_ila_simulator(self):
    print('\n--------------------------------------------------------------')
    print('\tinvoking ILA simulator')
    print('--------------------------------------------------------------\n')
    # measure the time of ila simulation
    cmd = ['hlscnn_asm_sim_driver', './test/conv_ila_prog_frag.json', './test/conv_ila_out.json']
    start_time = timeit.default_timer()
    subprocess.run(cmd)
    end_time = timeit.default_timer()
    print('\n********* ILA simulator performance ***********')
    print('ILA simulator execution time is {:04f}s'.format(end_time - start_time))
  
  def collect_ila_result(self):
    print('\n--------------------------------------------------------------')
    print('\tcollecting ILA simulation result')
    print('--------------------------------------------------------------\n')
    with open('./test/conv_ila_out.json', 'r') as fin:
      ila_out = json.load(fin)
      result = []
      for addr in self.out_addr:
        key = f'0x{addr:08X}'
        result.append(ila_out[key])
    np.asarray(result).tofile('./data/conv_result.txt', sep='\n')

  def run(self):
    subprocess.run(['mkdir', '-p', 'test', 'data'])
    self.collect_data_in()
    self.produce_asm_all()
    self.produce_prog_frag()
    self.invoke_ila_simulator()
    self.collect_ila_result()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='ConvLayer Parameters')
  parser.add_argument('--in_size', nargs='+', type=int)
  parser.add_argument('--out_size', nargs='+', type=int)
  parser.add_argument('--kernel_size', nargs='+', type=int)
  parser.add_argument('--stride', nargs='+', type=int)
  parser.add_argument('--is_bias', action='store_const', const=1, default=0)
  parser.add_argument('--bias', type=float, default=0)
  parser.add_argument('--is_relu', action='store_const', const=1, default=0)
  parser.add_argument('--is_accum', action='store_const', const=1, default=0)
  args = parser.parse_args()
  
  driver = conv_layer_driver(
    inp_size = args.in_size,
    out_size = args.out_size,
    kernel_size = args.kernel_size,
    stride = args.stride,
    is_bias = args.is_bias,
    bias = args.bias,
    is_relu = args.is_relu,
    is_accum = args.is_accum
  )
  driver.run()
  # driver.produce_conv_layer_asm()

  # attrs = vars(driver)
  # print(',\n '.join("%s: %s" % item for item in attrs.items()))




