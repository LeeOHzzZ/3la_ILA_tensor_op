import json
import os

from .utils import tool
from ._ts_asm_converter import ts_asm_converter as ts_cvtr
from ._asm_prog_frag_converter import asm_prog_frag_converter as prog_cvtr

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
  
  def to_ila_prog_frag(self):
    """
    convert ILA tensor assembly to flexnlp-ila program fragment
    """
    if not self.is_to_asm:
      self.to_ila_asm()
    cvtr = prog_cvtr(self.asm_list, self.data_lib)
    self.prog_frag = cvtr.to_ila_prog_frag()
    self.is_to_prog = True

  def to_axi_cmds(self, base_addr):
    """
    convert ILA prog fragment to flexnlp axi commands
      base_addr: the base addr of FlexNLP in the SoC
        0x33000000: this is for FlexNLP simulation
        base_addr: this is for driving FPGA
    """
    self.axi_cmd_list = []
    if not self.is_to_prog:
      self.to_ila_prog_frag
    for insn in self.prog_frag:
      mode = insn['mode']
      addr = insn['addr']
      if '0x33000000' not in base_addr:
        addr = hex(int(addr, base=16) - 0x33000000 + int(base_addr, base=16))
      data = insn['data']
      axi_cmd = '2,' + mode + ',' + addr + ',' + data + '\n'
      self.axi_cmd_list.append(axi_cmd)
    self.is_to_axi = True
  
  def to_test_vec_for_fpga(self, base_addr, op_name):
    """
    convert ILA prog fragment to test_vector accepted 
    by FlexNLP FPGA implementation
    based on the create_test_vec.pl provided by Thierry
    example: weight128.val64[1] = 0x0A97DED717F647FF; weight128.val64[0] = 0xFEF6EB776CEBF455; HW128_REG(0xa0500000) = weight128.val128;
    """
    self.test_vec_wr_list = []
    self.test_vec_rd_list = []

    if not self.is_to_prog:
      self.to_ila_prog_frag
    for insn in self.prog_frag:
      addr = insn['addr']
      if '0x33000000' not in base_addr:
        addr = hex(int(addr, base=16) - 0x33000000 + int(base_addr, base=16))
      mode = insn['mode']
      if mode == 'W':
        data = insn['data'][2:]
        # separate the higher and lower 64 bit data
        if len(data) > 16:
          data_l = '0x' + data[-16:]
          if len(data) > 32:
            data_h = '0x' + data[len(data)-32:16-len(data)+1]
          else:
            data_h = '0x' + data[0:len(data)-16]
        else:
          data_l = '0x' + data
          data_h = '0x0'
        self.test_vec_wr_list.append(
          'weight128.val64[1] = {}; weight128.val64[0] = {};HW128_REG({}) = weight128.val128;\n'.format(data_h, data_l, addr)
          )
      elif mode == 'R':
        self.test_vec_rd_list.append(
          'read_data.val128 = HW128_REG(' + addr + ');\n' + \
          'printf("[read_out_' + op_name + ']: {\\"0x%llX\\" : \\"0x%016llx%016llx\\"}\\n",' + addr + \
          ', read_data.val64[1], read_data.val64[0]);\n'
        )
  
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
  
  def dump_axi_cmds(self, out_path, flexnlp_base_addr = '0x33000000', op_name = ''):
    """
    dump axi_cmds to JSON file
    """
    if not self.is_to_axi:
      self.to_axi_cmds(flexnlp_base_addr)
      if os.environ.get('USE_3LA_FPGA') in ('1', 'ON'):
        self.to_test_vec_for_fpga(flexnlp_base_addr, op_name= op_name)
    with open(out_path, 'w') as fout:
      fout.writelines(self.axi_cmd_list)
    
    # get the code script template
    self.tl = tool()
    temp = self.tl.get_axi_cmd_template()

    with open('./test/fpga_axi_set_cmds.h', 'w') as fout:
      fout.write("#ifndef SET_AXI_CMDS_H_\n#define SET_AXI_CMDS_H_\n\n")
      fout.write(temp)
      fout.write('\nint set_axi_wr_cmds() {\n')
      fout.writelines(self.test_vec_wr_list)
      fout.writelines('\n}\n')
      fout.write('\nint set_axi_rd_cmds() {\n')
      fout.writelines(self.test_vec_rd_list)
      fout.write('\n}\n\n#endif\n')

  

 