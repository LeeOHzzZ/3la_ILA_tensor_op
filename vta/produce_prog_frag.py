from asm_ila_instr_converter import convert
import sys

if __name__ == "__main__":
  assert len(sys.argv) == 4, "incorrect arg number provided, need 4 instead of " \
                              + str(len(sys.argv))

  asm_path = sys.argv[1]
  data_path = sys.argv[2]
  dest_path = sys.argv[3]
  convert(asm_path, data_path, dest_path)
  