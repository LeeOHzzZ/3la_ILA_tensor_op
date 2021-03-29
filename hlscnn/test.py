import json

from converter import Converter as cvtr

"""
The Virtual SOC memory in ILA has an offset address of 0x50000
Thus, when setting ActBase as 0x10000, the vir_mem_wr for act
should start from 0x10000 + 0x50000 = 0x60000
"""


def gen_test_asm(out_path):
  asm_list = []
  
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x50000',
    'data' : '0x00000000000000160000000000000013'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x50010',
    'data' : '0x00000000000000210000000000000007'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x50020',
    'data' : '0x0000000000000017000000000000002f'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x50030',
    'data' : '0x00000000000000140000000000000000'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x50040',
    'data' : '0x00000000000000140000000000000027'
  })

  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60000',
    'data' : '0x0000000000000000000000000000001d'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60010',
    'data' : '0x00000000000000000000000000000069'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60020',
    'data' : '0x00000000000000000000000000000004'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60030',
    'data' : '0x00000000000000000000000000000058'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60040',
    'data' : '0x00000000000000000000000000000056'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60050',
    'data' : '0x00000000000000000000000000000073'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60060',
    'data' : '0x00000000000000000000000000000006'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60070',
    'data' : '0x00000000000000000000000000000077'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60080',
    'data' : '0x00000000000000000000000000000054'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x60090',
    'data' : '0x0000000000000000000000000000001b'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x600a0',
    'data' : '0x0000000000000000000000000000002c'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x600b0',
    'data' : '0x00000000000000000000000000000009'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x600c0',
    'data' : '0x0000000000000000000000000000004e'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x600d0',
    'data' : '0x0000000000000000000000000000005a'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x600e0',
    'data' : '0x00000000000000000000000000000010'
  })
  asm_list.append({
    'name' : 'VirMemWr',
    'addr' : '0x600f0',
    'data' : '0x0000000000000000000000000000007a'
  })

  asm_list.append({
    'name' : 'CfgSoCMemRdWrLen',
    'length' : 1
  })

  for i in range(5):
    asm_list.append({
      'name' : 'CfgSoCMemBaseAddr',
      'base_addr' : hex(0x00000 + i * 0x10)
    })
    asm_list.append({
      'name' : 'SpadWr',
      'addr' : hex(0x4000 + i*0x10)
    })

  asm_list.append({
    'name' : 'CfgConvActBaseAddr',
    'base_addr' : '0x00010000'
  })
  asm_list.append({
    'name' : 'CfgConvWgtBaseAddr',
    'base_addr' : '0x00004000'
  })
  asm_list.append({
    'name' : 'CfgConvOutBaseAddr',
    'base_addr' : '0x00024000'
  })
  asm_list.append({
    'name' : 'CfgConvInpSize',
    'inp_cols' : 0x4,
    'inp_rows' : 0x4,
    'inp_chans' : 0x1
  })
  asm_list.append({
    'name' : 'CfgConvOutSize',
    'out_cols' : 0x2,
    'out_rows' : 0x2,
    'out_chans' : 0x1
  })
  asm_list.append({
    'name' : 'CfgConvKernelSize',
    'kernel_cols' : 3,
    'kernel_rows' : 3,
    'kernel_c_stride' : 1,
    'kernel_r_stride' : 1 
  })
  asm_list.append({
    'name' : 'CfgConvChan',
    'chan_bias' : 0x226A,
    'is_bias' : 1,
    'is_relu' : 1,
    'is_accum' : 0,
    'kernel_num' : 1,
    'is_wb' : 0
  })
  asm_list.append({
    'name' : 'ConvStart',
  })

  asm_list.append({
    'name' : 'SpadRd',
    'addr' : '0x24050'
  })
  asm_list.append({
    'name' : 'SpadRd',
    'addr' : '0x24060'
  })
  asm_list.append({
    'name' : 'SpadRd',
    'addr' : '0x24090'
  })
  asm_list.append({
    'name' : 'SpadRd',
    'addr' : '0x240a0'
  })

  with open(out_path, 'w') as fout:
    json.dump({'asm' : asm_list}, fout, indent=4)

def gen_test_data_lib(out_path):
  
  data_lib = []
  with open(out_path, 'w') as fout:
    json.dump(data_lib, fout, indent=4)

if __name__ == '__main__':
  gen_test_asm('test_asm.json')
  gen_test_data_lib('test_data_lib.json')

  prog_cvtr = cvtr('test_asm.json', 'test_data_lib.json')
  prog_cvtr.dump_ila_prog_frag('test_prog_frag_in.json')
