import json
import sys

def produce_asm(out_path):
  ila_asm = []
  # simple load_wgt test
  # load_wgt sram_id dram_id y_size x_size x_stride
  ila_asm.append({
    'name' : 'load_wgt',
    'sram_id' : 0,
    'dram_id' : 0,
    'y_size' : 1,
    'x_size' : 1,
    'x_stride' : 1
  })
  # simple load_inp test
  # assembly: load_inp sram_id dram_id y_size x_size x_stride y_pad0 y_pad1 x_pad0 x_pad1
  ila_asm.append({
    'name' : 'load_inp',
    'sram_id' : 1,
    'dram_id' : 2,
    'y_size' : 1,
    'x_size' : 8,
    'x_stride' : 16,
    'y_pad0' : 1,
    'y_pad1' : 1,
    'x_pad0' : 1,
    'x_pad1' : 1
  })
  # simple load_bias test
  # assembly: load_bias sram_id dram_id y_size x_size x_stride
  ila_asm.append({
    'name' : 'load_bias',
    'sram_id' : 1,
    'dram_id' : 2,
    'y_size' : 1,
    'x_size' : 8,
    'x_stride' : 16
  })
  # simple load_uop test
  # assembly: load_uop sram_id dram_id x_size
  ila_asm.append({
    'name' : 'load_uop',
    'sram_id' : 1,
    'dram_id' : 2,
    'x_size' : 8
  })
  # simple gemm test
  # assembly: gemm reset_f, uop_bgn, uop_end, iter_o, iter_i, dst_fo, dst_fi, src_fo, src_fi, wgt_fo, wgt_fi
  ila_asm.append({
    'name' : 'gemm',
    'reset_f' : 1,
    'uop_bgn' : 1,
    'uop_end' : 2,
    'iter_o' : 2,
    'iter_i' : 1,
    'dst_fo' : 1,
    'dst_fi' : 1,
    'src_fo' : 2,
    'src_fi' : 2,
    'wgt_fo' : 3,
    'wgt_fi' : 3
  })


  ila_asm = {'asm' : ila_asm}
  with open(out_path, 'w') as f:
    json.dump(ila_asm, f, indent=4)
  
  print('test assembly has been dumped to ' + out_path)

def produce_data(out_path):
  # placeholder for data
  data = []

  # sample data for different buffer
  data.append({
    'name' : 'input_buffer',
    'idx' : 1,
    'value' : '0x01'
  })

  data.append({
    'name' : 'weight_buffer',
    'idx' : 2,
    'value' : '0x02'
  })

  data.append({
    'name' : 'bias_buffer',
    'idx' : 3,
    'value' : '0x12345678'
  })

  data.append({
    'name' : 'uop_buffer',
    'idx' : 4,
    'value' : '0x0000111122223333'
  })

  data = {'data_dump' : data}
  with open(out_path, 'w') as f:
    json.dump(data, f, indent=4)
  
  print('test data has been dumped to ' + out_path)

if __name__ == "__main__":
  asm_out_path = 'asm_test.json'
  data_out_path = 'data_lib_test.json'
  if len(sys.argv) > 2:
    asm_out_path = sys.argv[1]
    data_out_path = sys.argv[2]

  produce_asm(asm_out_path)
  produce_data(data_out_path)
