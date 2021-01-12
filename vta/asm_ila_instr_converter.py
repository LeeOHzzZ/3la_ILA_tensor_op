import json
from vta_macros import *

instr_cntr = 1

def generate_vir_mem_insns(data):
  return []

def gen_load_wgt(asm):
  # load_wgt sram_id dram_id y_size x_size x_stride
  assert len(asm) == 6, 'wrong arguments for load_wgt'
  
  unused_bits = int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_WIDTH - 4 - VTA_MEMOP_ID_BITWIDTH - \
                VTA_MEMOP_SRAM_ADDR_BITWIDTH - VTA_MEMOP_DRAM_ADDR_BITWIDTH
  # use this awkward translation because bitwidth of mem_id is 2...
  bin_instr_l = unused_bits * '0' + \
                bin(asm['dram_id'])[2:].zfill(VTA_MEMOP_DRAM_ADDR_BITWIDTH) + \
                bin(asm['sram_id'])[2:].zfill(VTA_MEMOP_SRAM_ADDR_BITWIDTH) + \
                bin(VTA_MEM_ID_WGT)[2:].zfill(VTA_MEMOP_ID_BITWIDTH) + 4*'0' + \
                bin(VTA_OPCODE_LOAD)[2:].zfill(VTA_OPCODE_WIDTH)
  bin_instr_h = bin(asm['x_stride'])[2:].zfill(VTA_MEMOP_STRIDE_BITWIDTH) + \
                bin(asm['x_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH) + \
                bin(asm['y_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH)

  instr = hex(int(bin_instr_h, base=2)) + \
          hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  
  global instr_cntr
  ret = [{
    'instr_No.' : instr_cntr,
    'instr' : instr,
    'mem_bias_in' : '0',
    'mem_idx' : 0,
    'mem_mode' : 0,
    'mem_inp_in' : '0',
    'mem_uop_in' : '0',
    'mem_wgt_in' : '0',
  }]
  instr_cntr += 1
  return ret

def gen_load_inp(asm):
  return []

def gen_load_bias(asm):
  return []

def gen_load_uop(asm):
  return []

def gen_gemm(asm):
  return []

def generate_ila_insns(asm):
  asm_types = ['load_wgt', 'load_inp', 'load_bias', 'load_uop']
  asm_types += ['gemm']
  assert asm['name'] in asm_types, "not supported vta-ila assembly"

  # asm format: asm_name arg_0 [, arg_1, ...]
  return {
    'load_wgt' : gen_load_wgt(asm),
    'load_inp' : gen_load_inp(asm),
    'load_bias' : gen_load_bias(asm),
    'load_uop' : gen_load_uop(asm),
    'gemm' : gen_gemm(asm)
  } [asm['name']]

def convert_ila_insns(asm_dump, data_dump):
  """
  convert ILA assembly instructions into vta-ila instructions
  it has two parts:
    1. generate vir_mem_store instructions to set up the ILA memory
    2. convert vta-ila assembly to vta-ila instructions
  
  Parameters
  ----------

  """
  asm_dumps = asm_dump["asm"]
  data_dumps = data_dump["dump"]

  ret = []
  for v in data_dumps:
    ret += generate_vir_mem_insns(v)
  for asm in asm_dumps:
    ret += generate_ila_insns(asm)
  
  return ret
  
def convert(asm_path, data_path, dest_path):
  """
  Converts vta-ila assembly code and data into 
  corresponding vta-ila program fragment

  Parameters
  ----------
  asm_path: str
    Path to the vta-ila assembly JSON dump
  data_path: str
    Path to the data JSON dump
  dest_path: str
    Path at which corresponding ILA program fragment
  """

  with open(asm_path, 'r') as asm_f:
    asm_source = json.load(asm_f)
  with open(data_path, 'r') as data_f:
    data_source = json.load(data_f)
  
  ila_insns = convert_ila_insns(asm_source, data_source)
  prog_frag = {'program fragment' : ila_insns}

  with open(dest_path, 'w') as fout:
    json.dump(prog_frag, fout, indent=4)