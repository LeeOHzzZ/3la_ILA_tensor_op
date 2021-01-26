import json
import sys

def convert_axi_cmds(prog_frag):
  cmd_list = []
  for insn in prog_frag:
    mode = insn['mode']
    addr = insn['addr']
    data = insn['data']
    axi_cmd = '2,' + mode + ',' + addr + ',' + data + '\n'
    cmd_list.append(axi_cmd)
  return cmd_list


if __name__ == "__main__":
  assert len(sys.argv) == 3, "Usage: python3 gen_axi_commands.py [in_path] [out_path]"
  
  in_path = sys.argv[1]
  out_path = sys.argv[2]

  with open(in_path, 'r') as fin:
    prog_frag = json.load(fin)

  cmd_list = convert_axi_cmds(prog_frag['program fragment'])

  with open(out_path, 'w') as fout:
    fout.writelines(cmd_list)

  print("axi_commands file has been dumped to " + out_path)
