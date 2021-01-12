import json
import sys

def produce_asm(out_path):
  ila_asm = []
  # simple load_wgt test
  ila_asm.append({
    'name' : 'load_wgt',
    'sram_id' : 0,
    'dram_id' : 0,
    'y_size' : 1,
    'x_size' : 1,
    'x_stride' : 1
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