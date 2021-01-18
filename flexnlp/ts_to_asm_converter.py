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

def get_gb_large_v_addr(idx, num_vector, vector_idx):
  # calculate the start address offset in the gbcore large buffer
  timestep_size = num_vector * FLEXNLP_VECTOR_SIZE
  group_size = timestep_size * FLEXNLP_GBCORE_NUM_BANKS
  gb_buf_row_size = FLEXNLP_GBCORE_NUM_BANKS * FLEXNLP_VECTOR_SIZE
  out = (int(idx/FLEXNLP_GBCORE_NUM_BANKS)) * group_size + \
        (idx%FLEXNLP_GBCORE_NUM_BANKS) * FLEXNLP_VECTOR_SIZE + \
        gb_buf_row_size * vector_idx
  out += int(FLEXNLP_GB_LARGE_BUF_BASE, base=16)
  return hex(out)

def gen_gbmm(asm, data_lib):
  # gen_gbmm format
  # gen_gbmm num_vector_in
  return [{
    'name' : 'cfg_mmngr_gb_large',
    'base_0' : '0x0',
    'num_v_0' : asm['num_vector_in']
  }]

def gen_store_act(asm, data_lib):
  # gen_store_act format: gen_store_act timestep_idx, idx
  # timestep_idx: index of the tensor place holder
  # idx: timestep index in the flexnlp GB large buffer
  num_vector_in = data_lib['gb_num_vector_in']
  tensor_idx = asm['timestep_idx']
  idx = asm['idx']
  ret = []

  for v in range(num_vector_in):
    v_name = tensor_idx + '.' + str(v)
    addr = get_gb_large_v_addr(idx, num_vector_in, v)
    ret.append({
      'name' : 'write_v',
      'vector_name' : v_name,
      'addr' : addr
    })
  return ret

def gen_maxp(asm, data_lib):
  # gen_maxp format: gen_maxp num_ts
  # perform maxpooling to the first num_ts timestep in GBCore large buffer
  ret = []

  ret.append({
    'name' : 'cfg_ly_reduce',
    'mode' : 0,
    'mem_idx' : 0,
    'num_v' : data_lib['gb_num_vector_in'],
    'num_ts' : asm['num_ts']
  })

  ret.append({
    'name' : 'start_ly_reduce'
  })

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
  Convert flexnlp-ila tensor assembly into flexnlp-ila assembly
  """

  with open(asm_path, 'r') as asm_f:
    asm_source = json.load(asm_f)
  
  with open(data_path, 'r') as data_f:
    data_source = json.load(data_f)

  asm_insns = convert_ila_insns(asm_source, data_source)
  asm_insns = {'asm' : asm_insns}
  
  with open(dest_path, 'w') as f:
    json.dump(asm_insns, f, indent=4)

  print('flexnlp-ila assembly has been dumped to ' + dest_path)
  
  
