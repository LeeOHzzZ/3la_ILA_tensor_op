import sys
import json

def gen_asm_load_uop(param):
  return [{
    'name' : 'load_uop',
    'arg_0' : int(param['sram_base'], base=16),
    'arg_1' : int(param['dram_base'], base=16),
    'arg_2' : int(param['x_size'], base=16)
  }]

def gen_asm_load_bias(param):
  return [{
    'name' : 'load_bias',
    'arg_0' : int(param['sram_base'], base=16),
    'arg_1' : int(param['dram_base'], base=16),
    'arg_2' : int(param['y_size'], base=16),
    'arg_3' : int(param['x_size'], base=16),
    'arg_4' : int(param['x_stride'], base=16)
  }]

def gen_asm_load_inp(param):
  return [{
    'name' : 'load_inp',
    'arg_0' : int(param['sram_base'], base=16),
    'arg_1' : int(param['dram_base'], base=16),
    'arg_2' : int(param['y_size'], base=16),
    'arg_3' : int(param['x_size'], base=16),
    'arg_4' : int(param['x_stride'], base=16),
    'arg_5' : int(param['y_pad_0'], base=16),
    'arg_6' : int(param['y_pad_1'], base=16),
    'arg_7' : int(param['x_pad_0'], base=16),
    'arg_8' : int(param['x_pad_1'], base=16)
  }]

def gen_asm_load_wgt(param):
  return [{
    'name' : 'load_wgt',
    'arg_0' : int(param['sram_base'], base=16),
    'arg_1' : int(param['dram_base'], base=16),
    'arg_2' : int(param['y_size'], base=16),
    'arg_3' : int(param['x_size'], base=16),
    'arg_4' : int(param['x_stride'], base=16)
  }]

def gen_asm_store_acc(param):
  return [{
    'name' : 'store_acc',
    'arg_0' : int(param['sram_base'], base=16),
    'arg_1' : int(param['dram_base'], base=16),
    'arg_2' : int(param['y_size'], base=16),
    'arg_3' : int(param['x_size'], base=16),
    'arg_4' : int(param['x_stride'], base=16)
  }]

def gen_asm_gemm(param):
  return [{
    'name' : 'gemm',
    'arg_0' : int(param['reset_reg'], base=16),
    'arg_1' : int(param['uop_bgn'], base=16), 
    'arg_2' : int(param['uop_end'], base=16),
    'arg_3' : int(param['iter_out'], base=16),
    'arg_4' : int(param['iter_in'], base=16),
    'arg_5' : int(param['dst_factor_out'], base=16),
    'arg_6' : int(param['dst_factor_in'], base=16),
    'arg_7' : int(param['src_factor_out'], base=16),
    'arg_8' : int(param['src_factor_in'], base=16),
    'arg_9' : int(param['wgt_factor_out'], base=16),
    'arg_10' : int(param['wgt_factor_in'], base=16)
  }]

def parse_instr_log(instr_log):
  ret = []
  for i in instr_log['insns']:
    param = i['field_bytes']
    if 'load_uop' in param['asm_name']:
      ret += gen_asm_load_uop(param)
    if 'load_bias' in param['asm_name']:
      ret += gen_asm_load_bias(param)
    if 'load_inp' in param['asm_name']:
      ret += gen_asm_load_inp(param)
    if 'load_wgt' in param['asm_name']:
      ret += gen_asm_load_wgt(param)
    if 'store_acc' in param['asm_name']:
      ret += gen_asm_store_acc(param)
    if 'gemm' in param['asm_name']:
      ret += gen_asm_gemm(param)

  return ret

if __name__ == "__main__":
  assert len(sys.argv) == 3, "Usage: ./python3 asm_gen.py [src_path] [dest_path]"

  src_path = sys.argv[1]
  dest_path = sys.argv[2]

  with open(src_path, 'r') as fin:
    instr_log = json.load(fin)
  
  asm_list = {'asm' : parse_instr_log(instr_log)}

  with open(dest_path, 'w') as fout:
    json.dump(asm_list, fout, indent=4)