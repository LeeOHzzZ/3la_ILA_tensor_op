"""
  This file contains converter from hlscnn-ila assembly to
  hlscnn program fragments
"""

class asm_prog_frag_converter:
  __HLSCNN_BASE_ADDR = 0x32000000
  __instr_cntr = 1

  def __init__(self, asm_list, data_lib):
    self.asm_list = asm_list
    self.data_lib = data_lib
  
  def to_ila_prog_frag(self):
    self.prog_frag = []
    for asm in self.asm_list:
      self.prog_frag += self.generate_ila_prog_frag(asm)
    return self.prog_frag
  
  def generate_ila_prog_frag(self, asm):
    asm_types = ['SpadWr', 'VirMemWr', 'SpadRd']
    asm_types += ['CfgSoCMemBaseAddr', 'CfgSoCMemRdWrLen']
    asm_types += ['CfgConvActBaseAddr', 'CfgConvWgtBaseAddr', 'CfgConvOutBaseAddr']
    asm_types += ['CfgConvInpSize', 'CfgConvOutSize', 'CfgConvKernelSize', 'CfgConvChan']
    asm_types += ['ConvStart']

    assert asm['name'] in asm_types, \
      '{} is not a supported hlscnn-ila assembly'.format(asm['name'])
    
    if asm['name'] == 'SpadWr':
      return self.__gen_spad_write(asm)
    if asm['name'] == 'SpadRd':
      return self.__gen_spad_read(asm)
    if asm['name'] == 'VirMemWr':
      return self.__gen_vir_mem_write(asm)
    if asm['name'] == 'CfgSoCMemBaseAddr':
      return self.__gen_cfg_soc_mem_base_addr(asm)
    if asm['name'] == 'CfgSoCMemRdWrLen':
      return self.__gen_cfg_soc_mem_rd_wr_len(asm)
    if asm['name'] == 'CfgConvActBaseAddr':
      return self.__gen_cfg_conv_act_base_addr(asm)
    if asm['name'] == 'CfgConvWgtBaseAddr':
      return self.__gen_cfg_conv_wgt_base_addr(asm)
    if asm['name'] == 'CfgConvOutBaseAddr':
      return self.__gen_cfg_conv_out_base_addr(asm)
    if asm['name'] == 'CfgConvInpSize':
      return self.__gen_cfg_conv_input_size(asm)
    if asm['name'] == 'CfgConvOutSize':
      return self.__gen_cfg_conv_output_size(asm)
    if asm['name'] == 'CfgConvKernelSize':
      return self.__gen_cfg_kernel_size(asm)
    if asm['name'] == 'CfgConvChan':
      return self.__gen_cfg_conv_channel(asm)
    if asm['name'] == 'ConvStart':
      return self.__gen_conv_trigger(asm)
    
    raise NameError('Double check the asm name in the parser!')


  # ------------------------------------------------------------
  # ------------------------------------------------------------
  def __produce_insn(self, addr, data, mode, is_vir = 0):
    ret = [{
      'instr_No.' : self.__instr_cntr,
      'addr' : addr,
      'data' : data,
      'mode' : mode,
      'is_vir' : is_vir
    }]
    self.__instr_cntr += 1
    return ret

  """ 
  write SPAD instructions and virtual memory instruction
  """
  def __gen_spad_write(self, asm):
    # assembly: SpadWr [addr]
    # [addr]: hex string of SPAD address to write
    # SPAD0 address: 0x04000 ~ 0x24000 (20KB)
    # SPAD1 address: 0x24000 ~ 0x44000
    return self.__produce_insn(asm['addr'], '0x0', 'W')
  
  def __gen_spad_read(self, asm):
    # assembly: SpadRd [addr]
    # [addr]: hex string of SPAD address to read
    # SPAD0 address: 0x04000 ~ 0x24000 
    # SPAD1 address: 0x24000 ~ 0x44000
    return self.__produce_insn(asm['addr'], '0x0', 'R')

  def __gen_vir_mem_write(self, asm):
    # assembly: VirMemWr [addr], [data]
    # [addr]: hex string of virtual SoC memory address to write into
    # [data]: hex string of 128bit data
    return self.__produce_insn(asm['addr'], asm['data'], 'W', is_vir=1)

  """
  configuration instruction
  """
  """
  HLSCNN has a special data shift mechanism to acquire the configuration data
  from the AXI Slave input port (accel.h : 942)
  """
  def __gen_config_reg_addr(self, reg_id):
    """
    memory_map.h : 252
    """
    return hex(reg_id * 4 + self.__HLSCNN_BASE_ADDR)
  def __gen_config_data_offset(self, reg_id):
    return (reg_id * 4) % 16
  def __gen_binary_string(self, data, bitwidth):
    return format(data, '0{}b'.format(bitwidth))

  def __gen_cfg_soc_mem_base_addr(self, asm):
    # assembly: CfgSoCMemBaseAddr [base_addr]
    # [base_addr]: hex string of base address for reading from SoC memory
    # this value is set for reading the data from SoC memory through AXI master interface
    reg_id = 4
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    data = hex(int(asm['base_addr'], base=16)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_soc_mem_rd_wr_len(self, asm):
    # assembly: CfgSocMemRdWrLen [length]
    # [length]: int of rd/wr operation length
    reg_id = 5
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    data = hex(asm['length']) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')

  def __gen_cfg_conv_act_base_addr(self, asm):
    # assembly: CfgConvActBaseAddr [base_addr]
    # [base_addr]: hex string, the base address of the activation for convolution  
    
    # the register id for conv_act_base_addr in the design
    reg_id = 18
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    data = hex(int(asm['base_addr'], base=16)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_conv_wgt_base_addr(self, asm):
    # assembly: CfgConvWgtBaseAddr [base_addr]
    # [base_addr]: hex_string of base address of weights for convolution
    reg_id = 19
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    data = hex(int(asm['base_addr'], base=16)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_conv_out_base_addr(self, asm):
    # assembly: CfgConvOutBaseAddr [base_addr]
    # [base_addr]: hex_string of base address of outputs for convolution
    reg_id = 20
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    data = hex(int(asm['base_addr'], base=16)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_conv_input_size(self, asm):
    # assembly: CfgConvInpSize [inp_cols], [inp_rows], [inp_chans]
    # [inp_cols]: int of input columns of the convolution
    # [inp_rows]: int of input rows of the convolution
    # [inp_chans]: int of input channels of the convolution
    # // Layout of the AccelConvInputSizeConfig register.
    # //
    # // | Input channels | Input rows | Input cols |
    # // --------------------------------------------
    # // |      31-20     |    19-10   |     9-0    |
    # // --------------------------------------------
    reg_id = 21
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    cols = asm['inp_cols']
    rows = asm['inp_rows']
    chans = asm['inp_chans']
    data = hex(int((self.__gen_binary_string(chans, 12)
               + self.__gen_binary_string(rows, 10)
               + self.__gen_binary_string(cols, 10)),
               base=2)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_conv_output_size(self, asm):
    # assembly: CfgConvOutSize [out_cols], [out_rows], [out_chans]
    # [out_cols]: int of output columns of the convolution
    # [out_rows]: int of output rows of the convolution
    # [out_chans]: int of output channels of the convolution
    # // Layout of the AccelConvOutputSizeConfig register.
    # //
    # // | Output channels | Output rows | Output cols |
    # // -----------------------------------------------
    # // |       31-20     |     19-10   |     9-0     |
    # // -----------------------------------------------
    reg_id = 22
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    cols = asm['out_cols']
    rows = asm['out_rows']
    chans = asm['out_chans']
    data = hex(int(self.__gen_binary_string(chans, 12)
               + self.__gen_binary_string(rows, 10)
               + self.__gen_binary_string(cols,10),
               base = 2)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_kernel_size(self, asm):
    # assembly: CfgConvKernelSize [kernel_cols], [kernel_rows], [kernel_c_stride], [kernel_r_stride]
    # [kernel_cols]: int of kernel columns of the convolution
    # [kernel_rows]: int of kernel rows of the convolution
    # [kernel_c_stride]: int of kernel column stride
    # [kernel_r_stride]: int of kernel row stride
    # // Layout of the AccelKernelSizeConfig register.
    # //
    # // | Unused | Kernel Stride | Kernel rows | Kernel cols |
    # // ------------------------------------------------------
    # // |  31-24 |    21:16      |    15-8     |     7-0     |
    # // ------------------------------------------------------
    # kernel_c_stride is [18:16]
    # kernel_r_stride is [21:19]
    reg_id = 23
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    cols = asm['kernel_cols']
    rows = asm['kernel_rows']
    c_stride = asm['kernel_c_stride']
    r_stride = asm['kernel_r_stride']
    data = hex(int(self.__gen_binary_string(r_stride, 3)
               + self.__gen_binary_string(c_stride, 3)
               + self.__gen_binary_string(rows, 8)
               + self.__gen_binary_string(cols, 8),
               base = 2)) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
  
  def __gen_cfg_conv_channel(self, asm):
    # assembly: CfgConvChan [chan_bias], [is_bias], [is_relu], [is_accum], [kernel_num], [is_wb]
    # [chan_bias]: int of channel bias value
    # [is_bias]: int of whether enable bias on convolution
    # [is_relu]: int of whether enable relu on the convolution
    # [is_accum]: int of whether enable accumulation in the convolution
    # [kernel_num]: int of number of kernels/output channels of the convolution
    # [is_wb]: TODO: int of unclear operation
    # Warning: HLSCNN's document on this register is wrong!
    # // | Unused| ENABLE_WB |   FILTER_IDX        | ENABLE_ACCUMLATION_ON_OUTPUT_SPAD |ENABLE_RELU         |ENABLE_BIAS|CHANNEL_BIAS         |
    # //
    # // ------------------------------------------------------------------------------------------------------------------
    # // |  31-24|    22     |  21-19              |     18                            |         17         |     16   |     15-0            |
    # // ------------------------------------------------------------------------------------------------------------------

    reg_id = 24
    addr = self.__gen_config_reg_addr(reg_id)
    data_start_offset = self.__gen_config_data_offset(reg_id)
    chan_bias = asm['chan_bias']
    is_bias = asm['is_bias']
    is_relu = asm['is_relu']
    is_accum = asm['is_accum']
    kernel_num = asm['kernel_num']
    is_wb = asm['is_wb']
    data = hex(int(
      self.__gen_binary_string(is_wb, 1)
      + self.__gen_binary_string(kernel_num, 12)
      + self.__gen_binary_string(is_accum, 1)
      + self.__gen_binary_string(is_relu, 1)
      + self.__gen_binary_string(is_bias, 1)
      + self.__gen_binary_string(chan_bias, 16), base=2
    )) + data_start_offset*2*'0'
    return self.__produce_insn(addr, data, 'W')
    
  def __gen_conv_trigger(self, asm):
    # assembly: ConvStart
    # This assembly will trigger the convolution operation
    reg_id = 17
    addr = self.__gen_config_reg_addr(reg_id)
    return self.__produce_insn(addr, '0x0', 'W')
  


       
