import json

from _asm_prog_frag_converter import asm_prog_frag_converter 

class Converter:
  def __init__(self, asm_path, data_lib_path):
    # load asm and data_lib from files
    with open(asm_path, 'r') as fin:
      self.asm_list = json.load(fin)['asm']
    with open(data_lib_path, 'r') as fin:
      self.data_lib = json.load(fin)
    self.asm_cvtr = asm_prog_frag_converter(self.asm_list, self.data_lib)
    self.is_to_prog = False
    self.is_to_axi = False
  
  def to_ila_prog_frag(self):
    """
    convert ILA assembly to flexnlp-ila program fragment
    """
    self.prog_frag = self.asm_cvtr.to_ila_prog_frag()
    self.is_to_prog = True
  
  def to_axi_cmds(self):
    """
    convert to HLSCNN AXI commands
    """
    self.axi_cmd_list = []
    if not self.is_to_prog:
      self.to_ila_prog_frag()
    for insn in self.prog_frag:
      axi_cmd = '{},{},{}'.format(insn['mode'], insn['addr'], insn['data'])
      self.axi_cmd_list.append(axi_cmd)
    self.is_to_axi = True
  
  def dump_ila_prog_frag(self, out_path):
    """
    dump hlscnn-ila program fragment
    """
    if not self.is_to_prog:
      self.to_ila_prog_frag()
    with open(out_path, 'w') as fout:
      json.dump({'program fragment' : self.prog_frag}, fout, indent=4)
  
  def dump_axi_cmds(self, out_path):
    """
    dump HLSCNN AXI commands
    """
    if not self.is_to_axi:
      self.to_axi_cmds()
    with open(out_path, 'w') as fout:
      fout.writelines(self.axi_cmd_list)