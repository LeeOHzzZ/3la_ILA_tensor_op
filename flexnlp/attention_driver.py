import json
import sys
import numpy as np
import subprocess
import os
import argparse

import tvm

from src.utils import tool 
from src.converter import Converter as cvtr
from src.tool.standalone_attn import luong_general_attention

class attention_layer:
  def __init__(self, num_v, num_ts, mem_idx_enc, mem_idx_dec, op_name = 'attention'):
    """
    attention layer parameters
    num_v: number of vectors in each timestep
    num_ts: number of encoder timesteps corresponding to the decoder output
    mem_idx_enc: memory idx in the large buffer for encoder 
    mem_idx_dec: memory idx in the small buffer for the decoder
    """
    self.num_v = num_v
    self.num_ts = num_ts
    self.mem_idx_enc = mem_idx_enc
    self.mem_idx_dec = mem_idx_dec
    self.op_name = op_name
    self.tl = tool()

  def _produce_attention_asm(self):
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
      'adpbias_enc': int(self.adpbias_enc),
      'adpbias_dec': int(self.adpbias_dec),
      'adpbias_softmax': int(self.adpbias_softmax),
      'adpbias_out': int(self.adpbias_out)
    })
    # add instruction for reading the decoder output
    # the decoder output only has 1 timestep/tensor
    ila_asm.append({
      'name' : 'load_act',
      'mem_idx' : 0,
      'ts_idx' : 0,
      'mem_type' : 'small'
    })

    ila_asm = {'asm' : ila_asm}
    self.asm_out_dir = './test/attention_asm.json'
    with open(self.asm_out_dir, 'w') as fout:
      json.dump(ila_asm, fout, indent=4)
    print('*** ILA tensor assembly has been dumped to ', self.asm_out_dir)


  def _produce_data_lib(self):
    """
    This function produce the data library file
    """
    param = {
      'gb_num_vector_in' : self.num_v,
      'gb_num_vector_out' : self.num_v,
      'adpbias_enc' : int(self.adpbias_enc),
      'adpbias_dec' : int(self.adpbias_dec),
      'adpbias_softmax' : int(self.adpbias_softmax),
      'adpbias_out' : int(self.adpbias_out)
    }

    print('\n--------------------------------------------------------------')
    print('\tinvoking float to adpfloat converter')
    print('--------------------------------------------------------------\n')
    self.encoder_data.tofile('./test/enc_data_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/enc_data_q.tmp', self.adpbias_enc, './test/enc_data_q_av.tmp')
    self.decoder_data.tofile('./test/dec_data_q.tmp', sep = '\n')
    self.tl.call_float_adpt_v_cvtr('./test/dec_data_q.tmp', self.adpbias_dec, './test/dec_data_q_av.tmp')
    
    self._gen_data_lib_helper(param, "./test/attention_data_lib.json")

    return


  def _gen_data_lib_helper(self, param, out_path):
    self.data_lib = param
    # put adpfloat type input data into data-lib
    with open('./test/enc_data_q_av.tmp', 'r') as fin:
      enc_data_v_list = fin.read().splitlines()
    assert len(enc_data_v_list) == self.num_v * self.num_ts
    for t in range(self.num_ts):
      self.data_lib = (
        self.tl.vector_to_data_lib(enc_data_v_list[t*self.num_v : (t+1)*self.num_v],
                                f"ts_{t}", self.num_v, self.data_lib)
      )
    
    # put adpfloat type decoder vector into data-lib
    with open('./test/dec_data_q_av.tmp', 'r') as fin:
      dec_data_v_list = fin.read().splitlines()
    assert len(dec_data_v_list) == self.num_v
    self.data_lib = self.tl.vector_to_data_lib(dec_data_v_list, f"dec", self.num_v, self.data_lib)

    with open(out_path, 'w') as fout:
      json.dump(self.data_lib, fout, indent=4)
    print(f"\n*** data_lib has been dumped to {out_path}")

    return
  
  
  def _collect_data(self):
    print("""
    -------------------------------------------------------------------
    collecting input data from files
    -------------------------------------------------------------------
    """)
    dec_path = './data/dec.txt'
    print('collecting decoder data from', dec_path)
    self.decoder_data = np.fromfile(dec_path, sep='\n')
    enc_path = './data/enc.txt'
    print('collecting encoder data from', enc_path)
    self.encoder_data = np.fromfile(enc_path, sep='\n')

    batch_size = 1
    key_seq_len = self.num_ts
    #TODO: FlexNLP only suppot 1 decoder vector/timestep for each attention instruction
    # That's because once a context vector is generated, it needs to be fed into decoder
    # RNN for generating the next decoder hidden state?
    query_seq_len = 1 
    vector_size = 16*self.num_v # the size of each timestep
    # key is the encoder timesteps
    enc_shape = (batch_size, key_seq_len, vector_size)
    dec_shape = (batch_size, query_seq_len, vector_size)

    self.decoder_data.reshape(enc_shape)
    self.encoder_data.reshape(dec_shape)

    self.encoder_data, self.adpbias_enc = self.tl.get_adpfloat_bias(self.encoder_data)
    self.decoder_data, self.adpbias_dec = self.tl.get_adpfloat_bias(self.decoder_data)
    self.adpbias_enc += 10
    self.adpbias_dec += 10
    #TODO: set the adpbias for softmax and out to 2 for now
    self.adpbias_softmax = 2
    self.adpbias_out = 2

  # ---------------------------------------
  # invoke ILA simulation
  # ---------------------------------------
  def _gen_prog_frag(self):
    print('\n--------------------------------------------------------------')
    print('\tgenerate prog_frag.json for ILA simulator')
    print('--------------------------------------------------------------\n')
    self.ila_cvtr = cvtr(self.asm_out_dir, './test/attention_data_lib.json')
    self.ila_cvtr.dump_ila_asm('./test/attention_ila_asm.json')
    self.ila_cvtr.dump_ila_prog_frag('./test/attention_prog_frag_in.json')
    print('*** ILA program fragment has been dumped to ./test/attention_prog_frag_in.json***\n')
  
  
  def _collect_ila_result(self):
    """
    run ILA simulation and collect the result
    """
    self.result_ila = self.tl.collect_ila_result(
      in_path = "./test/attention_prog_frag_in.json",
      mem_idx = self.mem_idx_dec, num_ts = self.num_ts,
      num_vi = self.num_v, num_vo = self.num_v,
      bias = self.adpbias_out,
      mem_type = "small"
    )


  def run(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self._collect_data()
    self._produce_attention_asm()
    self._produce_data_lib()
    self._gen_prog_frag()
    self._collect_ila_result()
    self.result_ila.tofile('./data/result_attention_ila.txt', sep = '\n')
    self.tl.clean_up()
    

  def run_test(self):
    subprocess.run(['mkdir', '-p', 'npy', 'test', 'data'])
    self._get_ref_result()
    self._produce_attention_asm()
    self._produce_data_lib()
    self._gen_prog_frag()
    self._collect_ila_result()
    err_out_list = self._result_analysis(is_verbose=False)
    self.tl.clean_up()
    return err_out_list

  # -----------------------------------------
  # Function for testing 
  # -----------------------------------------
  def _get_ref_result(self):
    """
    Generate test data and get reference result from Relay implementation
    """
    batch_size = 1
    key_seq_len = self.num_ts
    #TODO: FlexNLP only suppot 1 decoder vector/timestep for each attention instruction
    # That's because once a context vector is generated, it needs to be fed into decoder
    # RNN for generating the next decoder hidden state?
    query_seq_len = 1 
    vector_size = 16*self.num_v # the size of each timestep
    # key is the encoder timesteps
    enc_shape = (batch_size, key_seq_len, vector_size)
    dec_shape = (batch_size, query_seq_len, vector_size)
    
    coef = 1
    self.encoder_data = coef * np.random.uniform(-1, 1, enc_shape)
    self.decoder_data = coef * np.random.uniform(-1, 1, dec_shape)
    
    test = coef * np.random.uniform(-1, 1, (10,))
    # quantize the data
    self.encoder_data, self.adpbias_enc = self.tl.get_adpfloat_bias(self.encoder_data)
    self.decoder_data, self.adpbias_dec = self.tl.get_adpfloat_bias(self.decoder_data)
    self.adpbias_enc += 10
    self.adpbias_dec += 10
    # single flexnlp attention instruction only support dot product, not support general 
    # luong attention, which would need BMM before that
    wgt_data = np.identity(vector_size)
    
    ref_cxt_vector = self.tl.get_relay_attention(
      key_seq_len, query_seq_len, vector_size, 
      self.encoder_data, self.decoder_data, wgt_data
    )

    self.ref_out = ref_cxt_vector
    self.adpbias_softmax = 2
    self.adpbias_out = self.tl.get_adpfloat_bias(ref_cxt_vector)[1] + 10

    return
  
  def _result_analysis(self, is_verbose):
    print('\n--------------------------------------------------------------')
    print('\tanalyze simulation result')
    print('--------------------------------------------------------------\n')
    err_out_list = []
    if not os.environ.get("USE_3LA_FPGA"):
      err_out = self.tl.cal_error_single_tensor(self.ref_out, self.result_ila)
      print("result: relative error (vs. ref): {:5.5%}\n".format(err_out))
    if is_verbose:
      print("reference output: \n{}\nresult: \n{}\n".format(self.ref_out, self.result_ila))
    err_out_list.append(err_out)

    return self.tl.cal_error_single_tensor(self.ref_out, self.result_ila)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='FlexASR Attention Layer Driver')
  parser.add_argument("--num_ts", type=int, required=True, help="number of timestep of the encoder")
  parser.add_argument("--num_v", type=int, required=True, help="number of vectors in each timestep")
  parser.add_argument("--mem_idx_enc", type=int, required=True, help="memory index of the encoder inputs")
  parser.add_argument("--mem_idx_dec", type=int, required=True, help="memory index of the decoder")
  parser.add_argument("--op_name", type=str, default="attention")
  kwargs = vars(parser.parse_args())
  driver = attention_layer(**kwargs)
  driver.run()
