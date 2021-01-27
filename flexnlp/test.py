import sys

import converter as cvtr

if __name__ == '__main__':
  assert len(sys.argv) == 4, \
    "Usage: python3 test.py [asm_path] [data_lib_path] [out_path]"
    
  
  cvtr = cvtr.Converter(sys.argv[1], sys.argv[2])
  # cvtr.to_ila_asm()
  # cvtr.dump_ila_asm(sys.argv[3])
  cvtr.dump_ila_prog_frag(sys.argv[3])
  cvtr.dump_axi_cmds(sys.argv[3])