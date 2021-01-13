import sys
import json

def parse_instr_log(instr_log):
  ret = []
  for i in range(len(instr_log)):
    if 'INSTRUCTION' in instr_log[i]:
      while 'instruction_end' not in instr_log[i]:
        


  return ret

if __name__ == "__main__":
  src_path = sys.argv[1]
  dest_path = sys.argv[2]

  with open(src_path, 'r') as fin:
    instr_log = fin.read().splitlines()
  
  asm_list = parse_instr_log(instr_log)

  with open(dest_path, 'w') as fout:
    json.dump(asm_list, fout, indent=4)