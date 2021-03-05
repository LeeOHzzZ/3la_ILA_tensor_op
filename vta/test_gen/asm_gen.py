import sys
import json

def gen_asm_load_uop(param):
  return [{
    'name' : 'load_uop',
    'sram_id' : int(param['sram_base'], base=16),
    'dram_id' : int(param['dram_base'], base=16),
    'x_size' : int(param['x_size'], base=16)
  }]

def gen_asm_load_bias(param):
  return [{
    'name' : 'load_bias',
    'sram_id' : int(param['sram_base'], base=16),
    'dram_id' : int(param['dram_base'], base=16),
    'y_size' : int(param['y_size'], base=16),
    'x_size' : int(param['x_size'], base=16),
    'x_stride' : int(param['x_stride'], base=16)
  }]

def gen_asm_load_inp(param):
  return [{
    'name' : 'load_inp',
    'sram_id' : int(param['sram_base'], base=16),
    'dram_id' : int(param['dram_base'], base=16),
    'y_size' : int(param['y_size'], base=16),
    'x_size' : int(param['x_size'], base=16),
    'x_stride' : int(param['x_stride'], base=16),
    'y_pad0' : int(param['y_pad_0'], base=16),
    'y_pad1' : int(param['y_pad_1'], base=16),
    'x_pad0' : int(param['x_pad_0'], base=16),
    'x_pad1' : int(param['x_pad_1'], base=16)
  }]

def gen_asm_load_wgt(param):
  return [{
    'name' : 'load_wgt',
    'sram_id' : int(param['sram_base'], base=16),
    'dram_id' : int(param['dram_base'], base=16),
    'y_size' : int(param['y_size'], base=16),
    'x_size' : int(param['x_size'], base=16),
    'x_stride' : int(param['x_stride'], base=16)
  }]

def gen_asm_store_acc(param):
  return [{
    'name' : 'store_acc',
    'sram_id' : int(param['sram_base'], base=16),
    'dram_id' : int(param['dram_base'], base=16),
    'y_size' : int(param['y_size'], base=16),
    'x_size' : int(param['x_size'], base=16),
    'x_stride' : int(param['x_stride'], base=16)
  }]

def gen_asm_gemm(param):
  return [{
    'name' : 'gemm',
    'reset_f' : int(param['reset_reg'], base=16),
    'uop_bgn' : int(param['uop_bgn'], base=16), 
    'uop_end' : int(param['uop_end'], base=16),
    'iter_o' : int(param['iter_out'], base=16),
    'iter_i' : int(param['iter_in'], base=16),
    'dst_fo' : int(param['dst_factor_out'], base=16),
    'dst_fi' : int(param['dst_factor_in'], base=16),
    'src_fo' : int(param['src_factor_out'], base=16),
    'src_fi' : int(param['src_factor_in'], base=16),
    'wgt_fo' : int(param['wgt_factor_out'], base=16),
    'wgt_fi' : int(param['wgt_factor_in'], base=16)
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

  print('assembly has been dumped to ' + dest_path)