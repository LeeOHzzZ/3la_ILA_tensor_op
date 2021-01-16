import json
import sys

def produce_asm(out_path):
  ila_asm = []
  # set num_vector_in
  ila_asm.append({
    'name' : 'set_gbmm',
    'num_vector_in' : 4
  })
  # some store instructions
  for i in range(4):
    ila_asm.append({
      'name' : 'store_act',
      'timestep_idx' : 'ts_' + str(i),
      'idx' : i
    })
  # maxp instruction
  ila_asm.append({
    'name' : 'maxp',
    'num_ts' : 4
  })
  
  ila_asm = {'asm': ila_asm}
  with open(out_path, 'w') as f:
    json.dump(ila_asm, f, indent=4)
  
  print("flexnlp-ila assembly has been dumped to " + out_path)

def produce_data(out_path):
  data_lib = {}
  
  # set up the num_vector
  data_lib['gb_num_vector_in'] = '0x4'
  # set up same random data
  data_lib['ts_0.0'] = '0x3C5ACB7A2CC234751CA3281B0231DB4'
  data_lib['ts_0.1'] = '0x0E4011C1A9032FBA813D3DC01A38A9AD'
  data_lib['ts_0.2'] = '0x0A2C2B4B5A9A04AAC3FD99843ABB93896'
  data_lib['ts_0.3'] = '0x37C9A3B512001ABE1431B59601B7BC96'

  data_lib['ts_1.0'] = '0x24D3B0D1B2CB33505FC4178149004ACB'
  data_lib['ts_1.1'] = '0x17404EC79A0A32C81C3F46A2284281B4'
  data_lib['ts_1.2'] = '0x0ACD1B5C0C5B64E1043E2B349B4B74727'
  data_lib['ts_1.3'] = '0x4CD199D31A9743DB2C87C08B01C7B8B0'

  data_lib['ts_2.0'] = '0x2FDDAFD9BEC93A5467B7A58159A354D3'
  data_lib['ts_2.1'] = '0x1B3756C91A1136CD28454DAA324633B5'
  data_lib['ts_2.2'] = '0x0B1D8AFB0D2C0523A46E9BE4EB4A1513C'
  data_lib['ts_2.3'] = '0x53D394D9200159E33335C3AF01CFA5C0'

  data_lib['ts_3.0'] = '0x34E2A1DBC6C132566DBCBC005DB04AD6'
  data_lib['ts_3.1'] = '0x295CC8351739D1244A52B8374CB3B5'
  data_lib['ts_3.2'] = '0x0B5DEA101D8B753454EF2C351AC185643'
  data_lib['ts_3.3'] = '0x54D18ADB1C2664E13B4BC5BB01CB00C4'

  with open(out_path, 'w') as f:
    json.dump(data_lib, f, indent=4)

  print('sample data file has been dumped to ' + out_path)

if __name__ == "__main__":
  asm_out_path = 'asm_test.json'
  data_out_path = 'data_lib_test.json'
  if len(sys.argv) > 2:
    asm_out_path = sys.argv[1]
    data_out_path = sys.argv[2]
  
  produce_asm(asm_out_path)
  produce_data(data_out_path)
