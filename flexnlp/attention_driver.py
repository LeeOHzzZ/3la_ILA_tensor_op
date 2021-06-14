import json
import sys
import numpy as np
import subprocess
import os
import argparse

from src.utils import tool
from src.converter import Converter as cvtr

class attention_layer:
  def __init__(self, num_v, num_ts, mem_idx_enc, mem_idx_dec,
               adpbias_enc, adpbias_dec, adpbias_softmax, adpbias_out,
               op_name = 'attention'):
    """
    attention layer parameters
    num_v: number of vectors in each timestep
    num_ts: number of encoder timesteps corresponding to the decoder output
    mem_idx_enc: memory idx in the large buffer for encoder 
    mem_idx_dec: memory idx in the small buffer for the decoder
    adpbias_enc: adaptive-float bias for encoder output matrix
    adpbias_dec: adaptive-float bias for input from the decoder
    adpbias_softmax: adaptive-float bias for softmax intermediate result
    adpbias_out: adaptive-float bias for attention output
    """
    self.num_v = num_v
    self.num_ts = num_ts
    self.mem_idx_enc = mem_idx_enc
    self.mem_idx_dec = mem_idx_dec
    self.adpbias_enc = adpbias_enc
    self.adpbias_dec = adpbias_dec
    self.adpbias_softmax = adpbias_softmax
    self.adpbias_out = adpbias_out
    self.op_name = op_name

  def produce_attention_asm(self):
    ila_asm = []
    # storing the encoder timestep into the large buffer
    for i in range(self.num_ts):
      ila_asm.append({
        'name' : 'store_act',
        'timestep_idx' : 'ts_' + str(i),
        'idx' : i,
        'mem_idx' : self.mem_idx_enc
      })
    # storing the decoder tensor into the small buffer
    ila_asm.append({
      'name' : 'store_dec',
      'idx' : 'dec'
    })
    # add instruction for attention
    ila_asm.append({
      'name' : 'attention',
      'num_ts' : self.num_ts,
      'num_v' : self.num_v,
      'mem_idx_enc': self.mem_idx_enc,
      'mem_idx_dec': self.mem_idx_dec,
      'adpbias_enc': self.adpbias_enc,
      'adpbias_dec': self.adpbias_dec,
      'adpbias_softmax': self.adpbias_softmax,
      'adpbias_out': self.adpbias_out
    })

    ila_asm = {'asm' : ila_asm}
    self.asm_out_dir = './test/attention_asm.json'
    with open(self.asm_out_dir, 'w') as fout:
      json.dump(ila_asm, fout, indent=4)
    print('*** ILA tensor assembly has been dumped to ', self.asm_out_dir)

  # ---------------------------------------
  # invoke ILA simulation
  # ---------------------------------------
  def gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr(self.asm_out_dir, './test/attention_data_lib.json')
    self.ila_cvtr.dump_ila_asm('./test/attention_ila_asm.json')
    # self.ila_cvtr.dump_ila_prog_frag('./test/attention_data_lib.json')
    # print('*** ILA program fragment has been dumped to ./test/attention_prog_frag_in.json***\n')
  
  def run(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self.produce_attention_asm()
    self.gen_prog_frag()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='FlexASR Attention Layer Driver')
  parser.add_argument("--num_ts", type=int, required=True, help="number of timestep of the encoder")
  parser.add_argument("--num_v", type=int, required=True, help="number of vectors in each timestep")
  parser.add_argument("--mem_idx_enc", type=int, required=True, help="memory index of the encoder inputs")
  parser.add_argument("--mem_idx_dec", type=int, required=True, help="memory index of the decoder")
  parser.add_argument("--adpbias_enc", type=int, required=True, help="adaptive-float bias for encoder output")
  parser.add_argument("--adpbias_dec", type=int, required=True, help="adaptive-float bias for decoder input")
  parser.add_argument("--adpbias_softmax", type=int, required=True,  
    help="adaptive-float bias for softmax intermediate result")
  parser.add_argument("--adpbias_out", type=int, required=True, 
    help="adaptive-float bias for the final outputs")
  parser.add_argument("--op_name", type=str, default="attention")
  kwargs = vars(parser.parse_args())
  driver = attention_layer(**kwargs)
  driver.run()
  

