"""
  This file contains converter functions from flexnlp-ila 
  assembly to flexnlp-ila program fragment
"""
class asm_prog_frag_converter:

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
    asm_types = ['write_v', 'read_v']
    asm_types += ['pe_cfg_rnn_layer_sizing', 'pe_cfg_mngr']
    asm_types += ['pe_cfg_act_mngr', 'pe_cfg_act_v']
    asm_types += ['cfg_mmngr_gb_large', 'cfg_mmngr_gb_small']
    asm_types += ['cfg_ly_reduce', 'cfg_gb_ctrl']
    asm_types += ['start']
    # wait for interrupt signal added for simulation
    asm_types += ['wait_irq']
      
    assert asm['name'] in asm_types, \
      "'" + asm['name'] + "' is not a supported flexnlp-ila assembly"

    if asm['name'] == 'write_v':
      return self.__gen_write_v(asm)
    if asm['name'] == 'read_v':
      return self.__gen_read_v(asm)
    
    # pe instructions
    if asm['name'] == 'pe_cfg_rnn_layer_sizing':
      return self.__gen_pe_cfg_rnn_layer_sizing(asm)
    if asm['name'] == 'pe_cfg_mngr':
      return self.__gen_pe_cfg_mngr(asm)
    if asm['name'] == 'pe_cfg_act_mngr':
      return self.__gen_pe_cfg_act_mngr(asm)
    if asm['name'] == 'pe_cfg_act_v':
      return self.__gen_pe_cfg_act_v(asm)
    
    # memory manager instructions
    if asm['name'] == 'cfg_mmngr_gb_large':
      return self.__gen_cfg_mmngr_gb_large(asm)
    if asm['name'] == 'cfg_mmngr_gb_small':
      return self.__gen_cfg_mmngr_gb_small(asm)

    # function configuration instructions
    if asm['name'] == 'cfg_ly_reduce':
      return self.__gen_cfg_ly_reduce(asm)
    if asm['name'] == 'cfg_gb_ctrl':
      return self.__gen_cfg_gb_ctrl(asm)

    # function trigger instructions
    if asm['name'] == 'start':
      return self.__gen_start(asm)

    # wait for interrupt signals for simulation
    if asm['name'] == 'wait_irq':
      return self.__produce_insn('0x0', '0x0', 'Q')

  # ---------------------------------------------------------
  # ---------------------------------------------------------

  def __produce_insn(self, addr, data, mode):
    ret = [{
      'instr_No.' : self.__instr_cntr,
      'addr' : addr,
      'data' : data,
      'mode' : mode
    }]
    self.__instr_cntr += 1
    return ret

  def __gen_write_v(self, asm):
    # assembly: write_v [vector_name], [addr]
    # assumption:
    #   1.physical address is given as string in assembly
    #   2.data in data_lib is aleady hex string
    return self.__produce_insn(asm['addr'], self.data_lib[asm['vector_name']], 'W')

  def __gen_read_v(self, asm):
    # assembly: read_v [addr]
    return self.__produce_insn(asm['addr'], '0x0', 'R')

  # -------------------------------------------
  # PE instructions
  # -------------------------------------------
  def __gen_pe_cfg_rnn_layer_sizing(self, asm):
    # assembly: pe_cfg_rnn_layer_sizing [pe_idx], [is_zero], [is_cluster], [is_bias], [num_mngr], [num_v_out]
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_rnn_layer_sizing'
    assert len(asm) == 7, 'incorrect arguments for pe_cfg_rnn_layer_sizing'
    addr = hex(0x34000000 + asm['pe_idx'] * 0x01000000 + 0x00400010)
    instr = "0x0" + \
            hex(asm['num_v_out'])[2:].zfill(2) + hex(asm['num_mngr'])[2:].zfill(2) + \
            hex(asm['is_bias'])[2:].zfill(2) + hex(asm['is_cluster'])[2:].zfill(2) + \
            hex(asm['is_zero'])[2:].zfill(2) + '01'
    return self.__produce_insn(addr, instr, 'W')

  def __gen_pe_cfg_mngr(self, asm):
    # assembly: pe_cfg_mngr [pe_idx], [mngr_idx], [is_zero], [adpbias_wgt], [adpbias_bias], [adpbias_inp], [num_v_in], [base_wgt], [base_bias], [base_inp]
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_mngr'
    assert asm['mngr_idx'] in range(1,3), 'not supported mngr_idx for pe_cfg_mngr'
    assert len(asm) == 11, "incorrect arguments for pe_cfg_mngr"
    addr = hex(0x34000000 + asm['pe_idx'] * 0x01000000 + 0x00400000 + asm['mngr_idx'] * 0x20)
    instr = '0x0' + hex(asm['base_inp'])[2:].zfill(4) + \
            hex(asm['base_bias'])[2:].zfill(4) + hex(asm['base_wgt'])[2:].zfill(4) + \
            hex(asm['num_v_in'])[2:].zfill(4) + \
            hex(asm['adpbias_inp'])[2:].zfill(2) + hex(asm['adpbias_bias'])[2:].zfill(2) + \
            hex(asm['adpbias_wgt'])[2:].zfill(2) + hex(asm['is_zero'])[2:].zfill(2)
    return self.__produce_insn(addr, instr, 'W')

  def __gen_pe_cfg_act_mngr(self, asm):
    # assembly: pe_cfg_act_mngr [pe_idx], [is_zero], [adpfloat_bias], [num_insn], [num_v_out], [buf_base], [out_base]
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_act_mngr'
    addr = hex(0x34000000 + asm['pe_idx']*0x01000000 + 0x00800010)
    instr = '0x0' + \
            hex(asm['out_base'])[2:].zfill(2) + hex(asm['buf_base'])[2:].zfill(4) + \
            hex(asm['num_v_out'])[2:].zfill(4) + hex(asm['num_insn'])[2:].zfill(2) + \
            hex(asm['adpfloat_bias'])[2:].zfill(2) + hex(asm['is_zero'])[2:].zfill(2) + '01'
    return self.__produce_insn(addr, instr, 'W')

  def __gen_pe_cfg_act_v(self, asm):
    # assembly: pe_cfg_act_v [pe_idx], [v_idx], [insn_0], ..., [insn_15]
    # assumption: insn are all hex string
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_act_v'
    assert asm['v_idx'] in range(1,3), 'not supported v_idx for gen_pe_cfg_act_v'
    addr = hex(0x34000000 + asm['pe_idx']*0x01000000 + 0x00800000 + 0x10*(asm['v_idx']+1))
    instr = ''
    for i in range(16):
      key = 'insn_' + str(i)
      if key in asm:
        instr = asm[key][2:].zfill(2) + instr
      else:
        instr = 2*'0' + instr
    return self.__produce_insn(addr, '0x0'+instr, 'W')

  # -------------------------------------------
  # memory manager configuration instructions
  # -------------------------------------------
  def __gen_cfg_mmngr_gb_large(self, asm):
    # assemlby: cfg_mmngr_gb_large [base_0], [num_v_0] (, [base_1], [num_v_1], ..., [base_3], [num_v_3])
    # assumption:
    #   1. At least base_0 and num_v_0 are given, the rest are optional
    #   2. base address values are given as hex string
    #   3. num_vector values are given as integer
    addr = '0x33400010'
    data = ''
    for i in range(4):
      key_v = 'num_v_'+str(i)
      key_b = 'base_'+str(i)
      
      if key_v in asm:
        data = hex(asm[key_v])[2:].zfill(2) + data
      else:
        data = 2*'0' + data
      
      data = 2*'0' + data

      if key_b in asm:
        data = asm[key_b][2:].zfill(4) + data
      else:
        data = 4*'0' + data
    return self.__produce_insn(addr, '0x'+data, 'W')

  def __gen_cfg_mmngr_gb_small(self, asm):
    # assembly: cfg_mmgnr_gb_small [base_0] (, [base_1], ..., [base_7])
    addr = '0x33400020'
    data = ''
    for i in range(8):
      key_b = 'base_' + str(i)
      if key_b in asm:
        data = asm[key_b][2:].zfill(4) + data
      else:
        data = 4*'0' + data
    return self.__produce_insn(addr, '0x'+data, 'W')

  # -------------------------------------------
  # function configuration instructions
  # -------------------------------------------
  def __gen_cfg_ly_reduce(self, asm):
    # assembly: cfg_ly_reduce [mode], [mem_idx], [num_v], [num_ts]
    # assumptions:
    #   1. mode : int
    #   2. mem_idx : int
    #   3. num_v : int
    #   4. num_ts : int
    addr = '0x33800010'
    num_v_field = hex(asm['num_v'])[2:].zfill(4)
    mem_idx_field = hex(asm['mem_idx'])[2:].zfill(4)
    mode_field = hex(asm['mode'])[2:].zfill(2)
    valid_field = '01'
    instr = hex(asm['num_ts']) + num_v_field + mem_idx_field + 4*'0' + mode_field + valid_field

    return self.__produce_insn(addr, instr, 'W') 

  def __gen_cfg_gb_ctrl(self, asm):
    # assembly: cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]
    # assumptions: all input types are integer
    addr = '0x33700010'
    num_timestep = hex(asm['num_ts'])[2:].zfill(4)
    num_v_field = hex(asm['num_v_o'])[2:].zfill(2) + hex(asm['num_v_i'])[2:].zfill(2)
    mem_id_field = hex(asm['mem_id_o'])[2:].zfill(2) + hex(asm['mem_id_i'])[2:].zfill(2)
    rnn_flag_field = hex(asm['is_rnn'])[2:].zfill(4)
    mode_field = hex(asm['mode'])[2:].zfill(2)
    valid_field = '01'
    instr = '0x0' + num_timestep + num_v_field + mem_id_field + rnn_flag_field + mode_field + valid_field

    return self.__produce_insn(addr, instr, 'W')

  # -------------------------------------------
  # function trigger instructions
  # -------------------------------------------
  def __gen_start(self, asm):
    # assembly: start [op]
    assert asm['op'] in (1,2,3,4,5), "unsupported op function trigger"
    addr = {
      1 : '0x33000010',
      2 : '0x33000020',
      3 : '0x33000030',
      4 : '0x33000040',
      5 : '0x33000050'
    }.get(asm['op'])
    return self.__produce_insn(addr, '0x01', 'W')




  