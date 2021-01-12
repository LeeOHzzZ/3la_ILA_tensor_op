import json
import sys

def produce_asm(out_path):
  ila_asm = []
  # simple load_wgt test
  # load_wgt sram_id dram_id y_size x_size x_stride
  ila_asm.append({
    'name' : 'load_wgt',
    'arg_0' : 0,
    'arg_1' : 0,
    'arg_2' : 1,
    'arg_3' : 1,
    'arg_4' : 1
  })
  # simple load_inp test
  # assembly: load_inp sram_id dram_id y_size x_size x_stride y_pad0 y_pad1 x_pad0 x_pad1
  ila_asm.append({
    'name' : 'load_inp',
    'arg_0' : 1,
    'arg_1' : 2,
    'arg_2' : 1,
    'arg_3' : 8,
    'arg_4' : 16,
    'arg_5' : 1,
    'arg_6' : 1,
    'arg_7' : 1,
    'arg_8' : 1
  })
  # simple load_bias test
  # assembly: load_bias sram_id dram_id y_size x_size x_stride
  ila_asm.append({
    'name' : 'load_bias',
    'arg_0' : 1,
    'arg_1' : 2,
    'arg_2' : 1,
    'arg_3' : 8,
    'arg_4' : 16
  })
  # simple load_uop test
  # assembly: load_uop sram_id dram_id x_size
  ila_asm.append({
    'name' : 'load_uop',
    'arg_0' : 1,
    'arg_1' : 2,
    'arg_2' : 8
  })
  # simple gemm test
  # assembly: gemm reset_f, uop_bgn, uop_end, iter_o, iter_i, dst_fo, dst_fi, src_fo, src_fi, wgt_fo, wgt_fi
  ila_asm.append({
    'name' : 'gemm',
    'arg_0' : 1,
    'arg_1' : 1,
    'arg_2' : 2,
    'arg_3' : 2,
    'arg_4' : 1,
    'arg_5' : 1,
    'arg_6' : 1,
    'arg_7' : 2,
    'arg_8' : 2,
    'arg_9' : 3,
    'arg_10' : 3
  })


  ila_asm = {'asm' : ila_asm}
  with open(out_path, 'w') as f:
    json.dump(ila_asm, f, indent=4)

def produce_data(out_path):
  # placeholder for data
  data = ['data']

  data = {'dump' : data}
  with open(out_path, 'w') as f:
    json.dump(data, f, indent=4)

if __name__ == "__main__":
  asm_out_path = 'asm_test.json'
  data_out_path = 'data_lib_test.json'
  if len(sys.argv) > 2:
    asm_out_path = sys.argv[1]
    data_out_path = sys.argv[2]

  produce_asm(asm_out_path)
  produce_data(data_out_path)