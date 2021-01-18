import json

"""
Current supported flexnlp-ila assembly:

1. set_gbmm [num_vector_in]
    set up gb_memory_manager. Set up number of vectors in each input timestep
2. store_act [ts_idx], [idx]
    store activation/timestep [ts_idx] into entry index [idx] of gbcore large buffer
3. maxp [num_ts]
    perform maxpooling to the first [num_ts] timesteps in the gbcore large buffer
"""

FLEXNLP_VECTOR_SIZE = 16
FLEXNLP_GBCORE_NUM_BANKS = 16
FLEXNLP_GB_LARGE_BUF_BASE = '0x33500000'

instr_cntr = 1

def get_gb_large_ts_addr(idx, num_vector):
  # calculate the start address offset in the gbcore large buffer
  timestep_size = num_vector * FLEXNLP_VECTOR_SIZE
  group_size = timestep_size * FLEXNLP_GBCORE_NUM_BANKS
  out = (int(idx/FLEXNLP_GBCORE_NUM_BANKS)) * group_size + \
        (idx%FLEXNLP_GBCORE_NUM_BANKS) * FLEXNLP_VECTOR_SIZE
  return out

def gen_gbmm(asm, data_lib):
  # gen_gbmm format
  # gen_gbmm num_vector_in
  num_vector_in = asm['num_vector_in']
  addr = "0x33400010"
  data = hex(num_vector_in)
  mode = "W"
  global instr_cntr
  ret = [{
    'instr_No.' : instr_cntr,
    'addr' : addr,
    'data' : data,
    'mode' : mode
  }]

  instr_cntr += 1

  return ret

def gen_store_act(asm, data_lib):
  # gen_store_act format: gen_store_act timestep_idx, idx
  # timestep_idx: index of the tensor place holder
  # idx: timestep index in the flexnlp GB large buffer
  # num_vector_in = int(data_lib['gb_num_vector_in'], base=16)
  num_vector_in = data_lib['gb_num_vector_in']
  # assert num_vector_in is int, "gb_num_vector_in in data_lib is not int but " + str(type(num_vector_in))
 
  tensor_idx = asm['timestep_idx']
  # idx = int(asm['arg_1'], base = 16)
  idx = asm['idx']

  start_addr = get_gb_large_ts_addr(idx, num_vector_in)
  gb_buf_row_size = FLEXNLP_GBCORE_NUM_BANKS * FLEXNLP_VECTOR_SIZE
  global instr_cntr 
  ret = []
  for v in range(num_vector_in):
    vector_data = data_lib[tensor_idx + '.' + str(v)]
    addr = start_addr + v*gb_buf_row_size + int(FLEXNLP_GB_LARGE_BUF_BASE, base=16)
    ret.append({
      'instr_No.' : instr_cntr,
      'addr' : hex(addr),
      'data' : vector_data,
      'mode' : 'W'
    })
    instr_cntr += 1
  
  return ret

def gen_maxp(asm, data_lib):
  # gen_maxp format: gen_maxp num_ts
  # perform maxpooling to the first num_ts timestep in GBCore large buffer
  
  ret = []
  # first to setup the layerreduce configuration
  addr = '0x33800010'
  # num_vector_out == num_vector_in for maxpool
  num_vector_out = data_lib['gb_num_vector_in']
  num_timestep = asm['num_ts']
  # num_timestep = int(asm['arg_0'], base=16) \
  #                if isinstance(asm['arg_0'], str) and asm['arg_0'].startswith('0x') \
  #                else int(asm['arg_0'])
  config_instr = hex(num_timestep) + num_vector_out[2:].zfill(4) + '1'.zfill(12)
  global instr_cntr
  ret.append({
    'instr_No.' : instr_cntr,
    'addr' : addr,
    'data' : config_instr,
    'mode' : 'W'
  })

  instr_cntr += 1

  # second to trigger the maxpooling instruction
  addr = '0x33000020'
  ret.append({
    'instr_No.' : instr_cntr,
    'addr' : addr,
    'data' : '0x0',
    'mode' : 'W'
  })

  instr_cntr += 1

  return ret

def generate_ila_insns(asm, data_lib):
  """
  generate flexnlp-ila instructions from given asm and
  data library

  Parameters
  ----------
  asm: 
    JSON dump of current assembly code
  data_lib:
    JSON dump of data
  
  Return
  ------
  List of ILA instructions corresponding to current
  ILA assembly
  """

  asm_types = ["set_gbmm", "store_act", "maxp"]
  assert asm['name'] in asm_types, "not supported ILA assembly"

  # asm format: asm_name arg_0 [, arg_1 ...]
  if asm['name'] == "set_gbmm":
    return gen_gbmm(asm, data_lib)
  if asm['name'] == "store_act":
    return gen_store_act(asm, data_lib)
  if asm['name'] == "maxp":
    return gen_maxp(asm, data_lib)


def convert_ila_insns(asm_dump, data_dump):
  """
  convert ILA assembly instructions into flexnlp-ila instructions

  Parameters
  ----------
  asm_dump: dict[str, any]
    JSON dump of assembly codes
  data_dump: dict[str, str]
    JSON dump of data for the assembly to generate
    ILA instructions
  
  Return
  ------
  List of ILA instructions

  """
  asm_dumps = asm_dump["asm"]
  data_dumps = data_dump
  ret = []

  for asm in asm_dumps:
    ret = ret + (generate_ila_insns(asm, data_dumps))
  
  return ret
  
def convert(asm_path, data_path, dest_path):
  """
  Converts flexnlp-ila tensor level assembly and data into 
  corresponding flexnlp-ila program fragment.

  Parameters
  ----------
  asm_path: str
    Path to the flexnlp-ila assembly JSON dump
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

  prog_frag = {'program fragment': ila_insns}
  
  with open(dest_path, 'w') as f:
    json.dump(prog_frag, f, indent=4)

  print('flexnlp-ila program fragment has been dumped to ' + dest_path)
  
  
