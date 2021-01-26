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
  # assembly: read_v [addr]
  return produce_insn(asm['addr'], '0x0', 'R')

# -------------------------------------------
# PE instructions
# -------------------------------------------
def gen_pe_cfg_rnn_layer_sizing(asm):
  # assembly: pe_cfg_rnn_layer_sizing [pe_idx], [is_zero], [is_cluster], [is_bias], [num_mngr], [num_v_out]
  assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_rnn_layer_sizing'
  assert len(asm) == 7, 'incorrect arguments for pe_cfg_rnn_layer_sizing'
  addr = hex(0x34000000 + asm['pe_idx'] * 0x01000000 + 0x00400010)
  instr = "0x0" + \
          hex(asm['num_v_out'])[2:].zfill(2) + hex(asm['num_mngr'])[2:].zfill(2) + \
          hex(asm['is_bias'])[2:].zfill(2) + hex(asm['is_cluster'])[2:].zfill(2) + \
          hex(asm['is_zero'])[2:].zfill(2) + '01'
  return produce_insn(addr, instr, 'W')

def gen_pe_cfg_mngr(asm):
  # assembly: pe_cfg_mngr [pe_idx], [mngr_idx], [is_zero], [adpbias_wgt], [adpbias_bias], [adpbias_inp], [num_v_in], [base_wgt], [base_bias], [base_inp]
  assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_mngr'
  assert asm['mngr_idx'] in range(1,3), 'not supported mngr_idx for pe_cfg_mngr'
  assert len(asm) == 11, "incorrect arguments for pe_cfg_mngr"
  addr = hex(0x34000000 + asm['pe_idx'] * 0x01000000 + 0x00400000 + asm['mngr_idx'] * 0x20)
  instr = '0x0' + hex(asm['base_inp'])[2:].zfill(4) + \
          hex(asm['base_bias'])[2:].zfill(4) + hex(asm['base_wgt'])[2:].zfill(4) + \
          hex(asm['num_v_in'])[2:].zfill(4) + \
          hex(asm['adpbias_inp'])[2:].zfill(2) + hex(asm['adpbias_bias'])[2:].zfill(2) + \
          hex(asm['adpbias_wgt'])[2:].zfill(2) + hex(asm['is_zero'])[2:].zfill(2)
  return produce_insn(addr, instr, 'W')

def gen_pe_cfg_act_mngr(asm):
  # assembly: pe_cfg_act_mngr [pe_idx], [is_zero], [adpfloat_bias], [num_insn], [num_v_out], [buf_base], [out_base]
  assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_act_mngr'
  addr = hex(0x34000000 + asm['pe_idx']*0x01000000 + 0x00800010)
  instr = '0x0' + \
          hex(asm['out_base'])[2:].zfill(2) + hex(asm['buf_base'])[2:].zfill(4) + \
          hex(asm['num_v_out'])[2:].zfill(4) + hex(asm['num_insn'])[2:].zfill(2) + \
          hex(asm['adpfloat_bias'])[2:].zfill(2) + hex(asm['is_zero'])[2:].zfill(2) + '01'
  return produce_insn(addr, instr, 'W')

def gen_pe_cfg_act_v(asm):
  # assembly: pe_cfg_act_v [pe_idx], [v_idx], [insn_0], ..., [insn_15]
  # assumption: insn are all hex string
  assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_act_v'
  assert asm['v_idx'] in range(1,3), 'not supported v_idx for gen_pe_cfg_act_v'
  addr = hex(0x34000000 + asm['pe_idx']*0x01000000 + 0x00800000 + 0x10*(asm['v_idx']+1))
  instr = ''
  for i in range(16):
    key = 'insn_' + str(i)
    if key in asm:
      instr = asm[key][2:].zfill(2) + instr
    else:
      instr = 2*'0' + instr
  return produce_insn(addr, '0x0'+instr, 'W')

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

def gen_cfg_gb_ctrl(asm):
  # assembly: cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]
  # assumptions: all input types are integer
  addr = '0x33700010'
  num_timestep = hex(asm['num_ts'])[2:].zfill(4)
  num_v_field = hex(asm['num_v_o'])[2:].zfill(2) + hex(asm['num_v_i'])[2:].zfill(2)
  mem_id_field = hex(asm['mem_id_o'])[2:].zfill(2) + hex(asm['mem_id_i'])[2:].zfill(2)
  rnn_flag_field = hex(asm['is_rnn'])[2:].zfill(4)
  mode_field = hex(asm['mode'])[2:].zfill(2)
  valid_field = '01'
  instr = '0x0' + num_timestep + num_v_field + mem_id_field + rnn_flag_field + mode_field + valid_field

  return produce_insn(addr, instr, 'W')

# -------------------------------------------
# function trigger instructions
# -------------------------------------------
def gen_start(asm):
  # assembly: start [op]
  assert asm['op'] in (1,2,3,4,5), "unsupported op function trigger"
  addr = {
    1 : '0x33000010',
    2 : '0x33000020',
    3 : '0x33000030',
    4 : '0x33000040',
    5 : '0x33000050'
  }.get(asm['op'])
  return produce_insn(addr, '0x01', 'W')

# --------------------------------------------
# --------------------------------------------
def generate_ila_insns(asm, data_lib):
  asm_types = ['write_v', 'read_v']
  asm_types += ['pe_cfg_rnn_layer_sizing', 'pe_cfg_mngr']
  asm_types += ['pe_cfg_act_mngr', 'pe_cfg_act_v']
  asm_types += ['cfg_mmngr_gb_large', 'cfg_mmngr_gb_small']
  asm_types += ['cfg_ly_reduce', 'cfg_gb_ctrl']
  asm_types += ['start']
  # wait for interrupt signal added for simulation
  asm_types += ['wait_irq']
    
  assert asm['name'] in asm_types, \
    "'" + asm['name'] + "' is not a supported flexnlp-ila assembly"

  if asm['name'] == 'write_v':
    return gen_write_v(asm, data_lib)
  if asm['name'] == 'read_v':
    return gen_read_v(asm)
  
  # pe instructions
  if asm['name'] == 'pe_cfg_rnn_layer_sizing':
    return gen_pe_cfg_rnn_layer_sizing(asm)
  if asm['name'] == 'pe_cfg_mngr':
    return gen_pe_cfg_mngr(asm)
  if asm['name'] == 'pe_cfg_act_mngr':
    return gen_pe_cfg_act_mngr(asm)
  if asm['name'] == 'pe_cfg_act_v':
    return gen_pe_cfg_act_v(asm)
  
  # memory manager instructions
  if asm['name'] == 'cfg_mmngr_gb_large':
    return gen_cfg_mmngr_gb_large(asm)
  if asm['name'] == 'cfg_mmngr_gb_small':
    return gen_cfg_mmngr_gb_small(asm)

  # function configuration instructions
  if asm['name'] == 'cfg_ly_reduce':
    return gen_cfg_ly_reduce(asm)
  if asm['name'] == 'cfg_gb_ctrl':
    return gen_cfg_gb_ctrl(asm)

  # function trigger instructions
  if asm['name'] == 'start':
    return gen_start(asm)

  # wait for interrupt signals for simulation
  if asm['name'] == 'wait_irq':
    return produce_insn('0x0', '0x0', 'Q')

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