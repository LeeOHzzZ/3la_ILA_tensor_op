import json

import tvm
from tvm.contrib.ly3la.flexnlp._ts_asm_converter import ts_asm_converter as ts_cvtr
from tvm.contrib.ly3la.flexnlp._asm_prog_frag_converter import asm_prog_frag_converter as prog_cvtr
# from _ts_asm_converter import ts_asm_converter as ts_cvtr
# from _asm_prog_frag_converter import asm_prog_frag_converter as prog_cvtr

class Converter:

  def __init__(self, asm_path, data_lib_path):
    # load asm and data_lib from files
    with open(asm_path, 'r') as fin:
      self.ts_asm_list = json.load(fin)['asm']
    with open(data_lib_path, 'r') as fin:
      self.data_lib = json.load(fin)
    self.is_to_asm = False
    self.is_to_prog = False
    self.is_to_axi = False
 
  def to_ila_asm(self):
    """
    convert ILA tensor assembly into flexnlp-ila assembly
    """
    cvtr = ts_cvtr(self.ts_asm_list, self.data_lib)
    self.asm_list = cvtr.to_ila_asm()
    self.is_to_asm = True
    # for asm in self.ts_asm_list:
    #   self.asm_list += (ts_cvtr.generate_ila_asm(self, asm))
  
  def to_ila_prog_frag(self):
    """
    convert ILA tensor assembly to flexnlp-ila program fragment
    """
    if not self.is_to_asm:
      self.to_ila_asm()
    cvtr = prog_cvtr(self.asm_list, self.data_lib)
    self.prog_frag = cvtr.to_ila_prog_frag()
    self.is_to_prog = True

  def to_axi_cmds(self):
    """
    convert ILA prog fragment to flexnlp axi commands
    """
    self.axi_cmd_list = []
    if not self.is_to_prog:
      self.to_ila_prog_frag
    for insn in self.prog_frag:
      mode = insn['mode']
      addr = insn['addr']
      data = insn['data']
      axi_cmd = '2,' + mode + ',' + addr + ',' + data + '\n'
      self.axi_cmd_list.append(axi_cmd)
    self.is_to_axi = True

  def dump_ila_asm(self, out_path):
    """
    dump flexnlp-ila assemlby to JSON file
    """
    if not self.is_to_asm:
      self.to_ila_asm()
    with open(out_path, 'w') as fout:
      json.dump(self.asm_list, fout, indent=4)
  
  def dump_ila_prog_frag(self, out_path):
    """
    dump flexnlp-ila program fragment to JSON file
    """
    if not self.is_to_prog:
      self.to_ila_prog_frag()
    with open(out_path, 'w') as fout:
      json.dump({'program fragment' : self.prog_frag}, fout, indent=4)
  
  def dump_axi_cmds(self, out_path):
    """
    dump axi_cmds to JSON file
    """
    if not self.is_to_axi:
      self.to_axi_cmds()
    with open(out_path, 'w') as fout:
      fout.writelines(self.axi_cmd_list)
  

 