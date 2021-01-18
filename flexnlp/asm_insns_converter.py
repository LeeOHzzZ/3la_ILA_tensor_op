import json

"""
This is the flexnlp-ila assembly to flexnlp-ila program fragment converter

Compilation flow:
flexnlp-ila tensor assembly -> flexnlp-ila assembly -> flexnlp-ila prog_frag

"""
instr_cntr = 1
def produce_insn(addr, data, mode):
  global instr_cntr
  ret = [{
    'instr_No.' : instr_cntr,
    'addr' : addr,
    'data' : data,
    'mode' : mode
  }]
  instr_cntr += 1
  return ret

def gen_write_v(asm, data_lib):
  # assembly: write_v [vector_name], [addr]
  # assumption:
  #   1.physical address is given as string in assembly
  #   2.data in data_lib is aleady hex string
  return produce_insn(asm['addr'], data_lib[asm['vector_name']], 'W')

def gen_read_v(asm):
  # assembly: read_v [vector_name], [addr]
  return produce_insn(asm['addr'], '0x0', 'R')

# -------------------------------------------
# memory manager configuration instructions
# -------------------------------------------
def gen_cfg_mmngr_gb_large(asm):
  # assemlby: cfg_mmngr_gb_large [base_0], [num_v_0] (, [base_1], [num_v_1], ..., [base_3], [num_v_3])
  # assumption:
  #   1. At least base_0 and num_v_0 are given, the rest are optional
  #   2. base address values are given as hex string
  #   3. num_vector values are given as integer
  addr = '0x33400010'
  data = ''
  for i in range(4):
    key_v = 'num_v_'+str(i)
    key_b = 'base_'+str(i)
    
    if key_v in asm:
      data = hex(asm[key_v])[2:].zfill(2) + data
    else:
      data = 2*'0' + data
    
    data = 2*'0' + data

    if key_b in asm:
      data = asm[key_b][2:].zfill(4) + data
    else:
      data = 4*'0' + data
  return produce_insn(addr, '0x'+data, 'W')

def gen_cfg_mmngr_gb_small(asm):
  # assembly: cfg_mmgnr_gb_small [base_0] (, [base_1], ..., [base_7])
  addr = '0x33400020'
  data = ''
  for i in range(8):
    key_b = 'base_' + str(i)
    if key_b in asm:
      data = asm[key_b][2:].zfill(4) + data
    else:
      data = 4*'0' + data
  return produce_insn(addr, '0x'+data, 'W')

# -------------------------------------------
# function configuration instructions
# -------------------------------------------
def gen_cfg_ly_reduce(asm):
  # assembly: cfg_ly_reduce [mode], [mem_idx], [num_v], [num_ts]
  # assumptions:
  #   1. mode : int
  #   2. mem_idx : int
  #   3. num_v : int
  #   4. num_ts : int
  addr = '0x33800010'
  num_v_field = hex(asm['num_v'])[2:].zfill(4)
  mem_idx_field = hex(asm['mem_idx'])[2:].zfill(4)
  mode_field = hex(asm['mode'])[2:].zfill(2)
  valid_field = '01'
  instr = hex(asm['num_ts']) + num_v_field + mem_idx_field + 4*'0' + mode_field + valid_field

  return produce_insn(addr, instr, 'W') 


# -------------------------------------------
# function trigger instructions
# -------------------------------------------
def gen_start_ly_reduces():
  # assembly: start_ly_reduce
  return produce_insn('0x33000020', '0x1', 'W')

def generate_ila_insns(asm, data_lib):
  asm_types = ['write_v', 'read_v']
  asm_types += ['cfg_mmngr_gb_large', 'cfg_mmngr_gb_small']
  asm_types += ['cfg_ly_reduce']
  asm_types += ['start_ly_reduce']
  assert asm['name'] in asm_types, \
    "'" + asm['name'] + "' is not a supported flexnlp-ila assembly"

  if asm['name'] == 'write_v':
    return gen_write_v(asm, data_lib)
  if asm['name'] == 'read_v':
    return gen_read_v(asm)
  
  # memory manager instructions
  if asm['name'] == 'cfg_mmngr_gb_large':
    return gen_cfg_mmngr_gb_large(asm)
  if asm['name'] == 'cfg_mmngr_gb_small':
    return gen_cfg_mmngr_gb_small(asm)

  # function configuration instructions
  if asm['name'] == 'cfg_ly_reduce':
    return gen_cfg_ly_reduce(asm)

  # function trigger instructions
  if asm['name'] == 'start_ly_reduce':
    return gen_start_ly_reduces()
  


def convert_ila_insns(asm_dump, data_dump):
  asm_list = asm_dump['asm']
  data_dict = data_dump
  ret = []

  for asm in asm_list:
    ret += generate_ila_insns(asm, data_dict)
  
  return ret


def convert(asm_path, data_path, dest_path):
  """
  Converts flexnlp-ila assembly and data into 
  corresponding flexnlp-ila program fragment
  """
  with open(asm_path, 'r') as asm_fin:
    asm_source = json.load(asm_fin)
  with open(data_path, 'r') as data_fin:
    data_source = json.load(data_fin)
  
  ila_insns = convert_ila_insns(asm_source, data_source)
  prog_frag = {'program fragment': ila_insns}

  with open(dest_path, 'w') as fout:
    json.dump(prog_frag, fout, indent=4)

  print('flexnlp-ila program fragment has been dumped to ' + dest_path)