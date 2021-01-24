# from ts_op_converter import convert
from ts_to_asm_converter import convert as ts_asm_converter
from asm_insns_converter import convert as asm_prog_frag_converter
import sys

if __name__ == '__main__':
  assert len(sys.argv) == 4, "Usage: python3 gen_prog_frag.py [asm_path] [data_path] [dest_path]"

  asm_path = sys.argv[1]
  data_path = sys.argv[2]
  dest_path = sys.argv[3]

  ila_asm_path = './test/intermediate_ila_asm.json'
  ts_asm_converter(asm_path, data_path, ila_asm_path)
  asm_prog_frag_converter(ila_asm_path, data_path, dest_path)