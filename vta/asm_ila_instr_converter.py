import json
from vta_macros import *

instr_cntr = 1

def prog_frag_gen(instr, mem_mode, mem_idx, data):
  global instr_cntr
  ret = [{
    'instr_No.' : instr_cntr,
    'instr_in' : instr,
    'mem_mode' : mem_mode,
    'mem_idx' : mem_idx,
    'mem_wgt_in' : data,
    'mem_inp_in' : data,
    'mem_bias_in' : data,
    'mem_uop_in' : data
  }]
  instr_cntr += 1
  return ret

def generate_vir_mem_insns(data):
  assert data['name'] in ['input_buffer', 'weight_buffer', 'bias_buffer', 'uop_buffer'], \
          'not supported data dump'
  if data['name'] == 'input_buffer':
    mem_mode = 1
  elif data['name'] == 'weight_buffer':
    mem_mode = 2
  elif data['name'] == 'bias_buffer':
    mem_mode = 3
  elif data['name'] == 'uop_buffer':
    mem_mode = 4

  return prog_frag_gen('0x0', mem_mode, data['idx'], data['value'])

def gen_load_wgt(asm):
  # load_wgt sram_id dram_id y_size x_size x_stride
  assert len(asm) == 6, 'wrong arguments for load_wgt'
  
  unused_bits = int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - VTA_MEMOP_ID_BITWIDTH - \
                VTA_MEMOP_SRAM_ADDR_BITWIDTH - VTA_MEMOP_DRAM_ADDR_BITWIDTH
  # use this awkward translation because bitwidth of mem_id is 2...
  bin_instr_l = unused_bits * '0' + \
                bin(asm['dram_id'])[2:].zfill(VTA_MEMOP_DRAM_ADDR_BITWIDTH) + \
                bin(asm['sram_id'])[2:].zfill(VTA_MEMOP_SRAM_ADDR_BITWIDTH) + \
                bin(VTA_MEM_ID_WGT)[2:].zfill(VTA_MEMOP_ID_BITWIDTH) + 4*'0' + \
                bin(VTA_OPCODE_LOAD)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_instr_h = bin(asm['x_stride'])[2:].zfill(VTA_MEMOP_STRIDE_BITWIDTH) + \
                bin(asm['x_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH) + \
                bin(asm['y_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH)

  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )

  return prog_frag_gen(instr, 0, 0, '0x0')

def gen_load_inp(asm):
  # assembly: load_inp sram_id dram_id y_size x_size x_stride y_pad0 y_pad1 x_pad0 x_pad1
  assert len(asm) == 10, "wrong arguments for load_inp: " + str(len(asm))

  unused_bits = int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - VTA_MEMOP_ID_BITWIDTH - \
                VTA_MEMOP_SRAM_ADDR_BITWIDTH - VTA_MEMOP_DRAM_ADDR_BITWIDTH
  
  bin_instr_l = unused_bits * '0' + \
                bin(asm['dram_id'])[2:].zfill(VTA_MEMOP_DRAM_ADDR_BITWIDTH) + \
                bin(asm['sram_id'])[2:].zfill(VTA_MEMOP_SRAM_ADDR_BITWIDTH) + \
                bin(VTA_MEM_ID_INP)[2:].zfill(VTA_MEMOP_ID_BITWIDTH) + 4*'0' + \
                bin(VTA_OPCODE_LOAD)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_instr_h = bin(asm['x_pad1'])[2:].zfill(VTA_MEMOP_PAD_BITWIDTH) + \
                bin(asm['x_pad0'])[2:].zfill(VTA_MEMOP_PAD_BITWIDTH) + \
                bin(asm['y_pad1'])[2:].zfill(VTA_MEMOP_PAD_BITWIDTH) + \
                bin(asm['y_pad0'])[2:].zfill(VTA_MEMOP_PAD_BITWIDTH) + \
                bin(asm['x_stride'])[2:].zfill(VTA_MEMOP_STRIDE_BITWIDTH) + \
                bin(asm['x_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH) + \
                bin(asm['y_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH)
  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )

  return prog_frag_gen(instr, 0, 0, '0x0')

def gen_load_bias(asm):
  # assembly: load_bias sram_id dram_id y_size x_size x_stride
  assert len(asm) == 6, 'wrong arguments for load_bias'
  
  unused_bits = int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - VTA_MEMOP_ID_BITWIDTH - \
                VTA_MEMOP_SRAM_ADDR_BITWIDTH - VTA_MEMOP_DRAM_ADDR_BITWIDTH
  # use this awkward translation because bitwidth of mem_id is 2...
  bin_instr_l = unused_bits * '0' + \
                bin(asm['dram_id'])[2:].zfill(VTA_MEMOP_DRAM_ADDR_BITWIDTH) + \
                bin(asm['sram_id'])[2:].zfill(VTA_MEMOP_SRAM_ADDR_BITWIDTH) + \
                bin(VTA_MEM_ID_ACC)[2:].zfill(VTA_MEMOP_ID_BITWIDTH) + 4*'0' + \
                bin(VTA_OPCODE_LOAD)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_instr_h = bin(asm['x_stride'])[2:].zfill(VTA_MEMOP_STRIDE_BITWIDTH) + \
                bin(asm['x_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH) + \
                bin(asm['y_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH)

  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )

  return prog_frag_gen(instr, 0, 0, '0x0')

def gen_load_uop(asm):
  # assembly format: load_uop sram_id dram_id x_size
  assert len(asm) == 4, 'wrong arguments for load_uop'
  unused_bits = int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - VTA_MEMOP_ID_BITWIDTH - \
                VTA_MEMOP_SRAM_ADDR_BITWIDTH - VTA_MEMOP_DRAM_ADDR_BITWIDTH
  
  bin_instr_l = unused_bits * '0' + \
                bin(asm['dram_id'])[2:].zfill(VTA_MEMOP_DRAM_ADDR_BITWIDTH) + \
                bin(asm['sram_id'])[2:].zfill(VTA_MEMOP_SRAM_ADDR_BITWIDTH) + \
                bin(VTA_MEM_ID_UOP)[2:].zfill(VTA_MEMOP_ID_BITWIDTH) + 4*'0' + \
                bin(VTA_OPCODE_LOAD)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_instr_h = bin(asm['x_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH) + \
                VTA_MEMOP_SIZE_BITWIDTH * '0'
  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )

  return prog_frag_gen(instr, 0, 0, '0x0')

def gen_store_acc(asm):
  # assembly format: store_acc sram_id, dram_id, y_size, x_size, x_stride
  assert len(asm) == 6, 'wrong arguments for store_acc'
  
  unused_bits = int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - VTA_MEMOP_ID_BITWIDTH - \
                VTA_MEMOP_SRAM_ADDR_BITWIDTH - VTA_MEMOP_DRAM_ADDR_BITWIDTH
  # use this awkward translation because bitwidth of mem_id is 2...
  bin_instr_l = unused_bits * '0' + \
                bin(asm['dram_id'])[2:].zfill(VTA_MEMOP_DRAM_ADDR_BITWIDTH) + \
                bin(asm['sram_id'])[2:].zfill(VTA_MEMOP_SRAM_ADDR_BITWIDTH) + \
                VTA_MEMOP_ID_BITWIDTH * '0' + 4 * '0' + \
                bin(VTA_OPCODE_STORE)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_instr_h = bin(asm['x_stride'])[2:].zfill(VTA_MEMOP_STRIDE_BITWIDTH) + \
                bin(asm['x_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH) + \
                bin(asm['y_size'])[2:].zfill(VTA_MEMOP_SIZE_BITWIDTH)

  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )

  return prog_frag_gen(instr, 0, 0, '0x0')

def gen_gemm(asm):
  # assembly format: gemm reset_f, uop_bgn, uop_end, iter_o, iter_i, dst_fo, dst_fi, src_fo, src_fi, wgt_fo, wgt_fi
  assert len(asm) == 12, 'wrong arguments for gemm'
  unused_bits = \
    (int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - 1 - VTA_GEMM_UOP_BEGIN_BITWIDTH - \
     VTA_GEMM_UOP_END_BITWIDTH - VTA_GEMM_ITER_OUT_BITWIDTH - VTA_GEMM_ITER_IN_BITWIDTH)
  
  bin_instr_l = unused_bits * '0' + \
                bin(asm['iter_i'])[2:].zfill(VTA_GEMM_ITER_IN_BITWIDTH) + \
                bin(asm['iter_o'])[2:].zfill(VTA_GEMM_ITER_OUT_BITWIDTH) + \
                bin(asm['uop_end'])[2:].zfill(VTA_GEMM_UOP_END_BITWIDTH) + \
                bin(asm['uop_bgn'])[2:].zfill(VTA_GEMM_UOP_BEGIN_BITWIDTH) + \
                bin(asm['reset_f'])[2:].zfill(1) + 4*'0' + \
                bin(VTA_OPCODE_GEMM)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_instr_h = bin(asm['wgt_fi'])[2:].zfill(VTA_GEMM_WGT_FACTOR_IN_BITWIDTH) + \
                bin(asm['wgt_fo'])[2:].zfill(VTA_GEMM_WGT_FACTOR_OUT_BITWIDTH) + \
                bin(asm['src_fi'])[2:].zfill(VTA_GEMM_SRC_FACTOR_IN_BITWIDTH) + \
                bin(asm['src_fo'])[2:].zfill(VTA_GEMM_SRC_FACTOR_OUT_BITWIDTH) + \
                bin(asm['dst_fi'])[2:].zfill(VTA_GEMM_DST_FACTOR_IN_BITWIDTH) + \
                bin(asm['dst_fo'])[2:].zfill(VTA_GEMM_DST_FACTOR_OUT_BITWIDTH)
  
  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )

  return prog_frag_gen(instr, 0, 0, '0x0')

def gen_alu(asm):
  # assembly format: alu_* reset_f, uop_bgn, uop_end, iter_o, iter_i, dst_fo, dst_fi, src_fo, src_fi, alu_op, use_imm, imm
  unused_bits = \
      (int(VTA_INSTR_BITWIDTH/2) - VTA_OPCODE_BITWIDTH - 4 - 1 - VTA_ALU_UOP_BEGIN_BITWIDTH - \
       VTA_ALU_UOP_END_BITWIDTH - VTA_ALU_ITER_OUT_BITWIDTH - VTA_ALU_ITER_IN_BITWIDTH)
  
  bin_instr_l = unused_bits * '0' + \
                bin(asm['iter_i'])[2:].zfill(VTA_GEMM_ITER_IN_BITWIDTH) + \
                bin(asm['iter_o'])[2:].zfill(VTA_GEMM_ITER_OUT_BITWIDTH) + \
                bin(asm['uop_end'])[2:].zfill(VTA_GEMM_UOP_END_BITWIDTH) + \
                bin(asm['uop_bgn'])[2:].zfill(VTA_GEMM_UOP_BEGIN_BITWIDTH) + \
                bin(asm['reset_f'])[2:].zfill(1) + 4*'0' + \
                bin(VTA_OPCODE_ALU)[2:].zfill(VTA_OPCODE_BITWIDTH)
  bin_imm = bin(((1 << VTA_ALU_IMM_BITWIDTH) - 1) & asm["imm"])
  bin_instr_h = (
    "0" + 
    bin_imm[2:].zfill(VTA_ALU_IMM_BITWIDTH) + 
    bin(asm['use_imm'])[2:].zfill(VTA_ALU_USE_IMM_FLAG_BITWIDTH) + 
    bin(asm['alu_op'])[2:].zfill(VTA_ALU_OPCODE_BITWIDTH) + 
    bin(asm['src_fi'])[2:].zfill(VTA_ALU_SRC_FACTOR_IN_BITWIDTH) + 
    bin(asm['src_fo'])[2:].zfill(VTA_ALU_SRC_FACTOR_OUT_BITWIDTH) + 
    bin(asm['dst_fi'])[2:].zfill(VTA_ALU_DST_FACTOR_IN_BITWIDTH) + 
    bin(asm['dst_fo'])[2:].zfill(VTA_ALU_DST_FACTOR_OUT_BITWIDTH)
  )
  instr = (
    "0x0" +
    hex(int(bin_instr_h, base=2))[2:] +
    hex(int(bin_instr_l, base=2))[2:].zfill(int(VTA_INSTR_BITWIDTH/8))
  )
  
  return prog_frag_gen(instr, 0, 0, '0x0')


def generate_ila_insns(asm):
  asm_types = ['load_wgt', 'load_inp', 'load_bias', 'load_uop', 'store_acc']
  asm_types += ['gemm']
  asm_types += ['alu_max', 'alu_min', 'alu_add', 'alu_shr', 'alu_mul']
  assert asm['name'] in asm_types, "not supported vta-ila assembly"

  # asm format: asm_name arg_0 [, arg_1, ...]
  if asm['name'] == 'load_wgt':
    return gen_load_wgt(asm)
  if asm['name'] == 'load_inp':
    return gen_load_inp(asm)
  if asm['name'] == 'load_bias':
    return gen_load_bias(asm)
  if asm['name'] == 'load_uop':
    return gen_load_uop(asm)
  if asm['name'] == 'store_acc':
    return gen_store_acc(asm)
  if asm['name'] == 'gemm':
    return gen_gemm(asm)
  if 'alu' in asm['name']:
    return gen_alu(asm)

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
  data_dumps = data_dump["data_dump"]

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

  print('ila program fragment has been dumped to ' + dest_path)