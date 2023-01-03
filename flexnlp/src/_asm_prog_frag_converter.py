"""
  This file contains converter functions from flexnlp-ila 
  assembly to flexnlp-ila program fragment
"""
class asm_prog_frag_converter:

  __FLEXNLP_BASE_ADDR = 0x33000000
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
    asm_types += ['cfg_ly_reduce', 'cfg_gb_ctrl', 'cfg_ly_norm', 'cfg_zeropadding', 'cfg_attention']
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
    if asm['name'] == 'cfg_ly_norm':
      return self.__gen_cfg_ly_norm(asm)
    if asm['name'] == 'cfg_zeropadding':
      return self.__gen_cfg_zeropadding(asm)
    if asm['name'] == 'cfg_attention':
      return self.__gen_cfg_attention(asm)

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
    addr = hex(self.__FLEXNLP_BASE_ADDR + (asm['pe_idx']+1)*0x01000000 + 0x00400010)
    instr = (
      "0x0" 
      + f"{asm['num_v_out']:0>2x}" 
      + f"{asm['num_mngr']:0>2x}" 
      + f"{asm['is_bias']:0>2x}"
      + f"{asm['is_cluster']:0>2x}"
      + f"{asm['is_zero']:0>2x}"
      + "01"
    )
    return self.__produce_insn(addr, instr, 'W')

  def __gen_pe_cfg_mngr(self, asm):
    # assembly: pe_cfg_mngr [pe_idx], [mngr_idx], [is_zero], [adpbias_wgt], [adpbias_bias], [adpbias_inp], [num_v_in], [base_wgt], [base_bias], [base_inp]
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_mngr'
    assert asm['mngr_idx'] in range(1,3), 'not supported mngr_idx for pe_cfg_mngr'
    assert len(asm) == 11, "incorrect arguments for pe_cfg_mngr"
    addr = hex(self.__FLEXNLP_BASE_ADDR + (asm['pe_idx']+1)*0x01000000 + 0x00400000 + asm['mngr_idx']*0x20)
    instr = (
      "0x0"
      + f"{asm['base_inp']:0>4x}"
      + f"{asm['base_bias']:0>4x}"
      + f"{asm['base_wgt']:0>4x}"
      + f"{asm['num_v_in']:0>4x}"
      + f"{asm['adpbias_inp']:0>2x}"
      + f"{asm['adpbias_bias']:0>2x}"
      + f"{asm['adpbias_wgt']:0>2x}"
      + f"{asm['is_zero']:0>2x}"
    )
    return self.__produce_insn(addr, instr, 'W')

  def __gen_pe_cfg_act_mngr(self, asm):
    # assembly: pe_cfg_act_mngr [pe_idx], [is_zero], [adpfloat_bias], [num_insn], [num_v_out], [buf_base], [out_base]
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_act_mngr'
    addr = hex(self.__FLEXNLP_BASE_ADDR + (asm['pe_idx']+1)*0x01000000 + 0x00800010)
    instr = (
      "0x0"
      + f"{asm['out_base']:0>2x}"
      + f"{asm['buf_base']:0>4x}"
      + f"{asm['num_v_out']:0>4x}"
      + f"{asm['num_insn']:0>2x}"
      + f"{asm['adpfloat_bias']:0>2x}"
      + f"{asm['is_zero']:0>2x}"
      + "01"
    )
    return self.__produce_insn(addr, instr, 'W')

  def __gen_pe_cfg_act_v(self, asm):
    # assembly: pe_cfg_act_v [pe_idx], [v_idx], [insn_0], ..., [insn_15]
    # assumption: insn are all hex string
    assert asm['pe_idx'] in range(4), 'not supported pe_idx for gen_pe_cfg_act_v'
    assert asm['v_idx'] in range(1,3), 'not supported v_idx for gen_pe_cfg_act_v'
    addr = hex(self.__FLEXNLP_BASE_ADDR + (asm['pe_idx']+1)*0x01000000 + 0x00800000 + 0x10*(asm['v_idx']+1))
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
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00400010)
    data = ''
    for i in range(4):
      key_v = 'num_v_'+str(i)
      key_b = 'base_'+str(i)
      if key_v in asm:
        data = f"{asm[key_v]:0>2x}" + data
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
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00400020)
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
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00800010)
    num_v_field = f"{asm['num_v']:0>4x}"
    mem_idx_field = f"{asm['mem_idx']:0>4x}"
    mode_field = f"{asm['mode']:0>2x}"
    valid_field = '01'
    instr = hex(asm['num_ts']) + num_v_field + mem_idx_field + 4*'0' + mode_field + valid_field

    return self.__produce_insn(addr, instr, 'W') 

  def __gen_cfg_ly_norm(self, asm):
    # assembly: cfg_ly_norm [mem_idx], [num_v], [num_ts], [adpbias_inp], [adpbias_beta], [adpbias_gamma]
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00900010)
    num_v_field = f"{asm['num_v']:0>4x}"
    mem_idx_field = f"{asm['mem_idx']:0>4x}"
    num_ts_field = f"{asm['num_ts']:0>8x}"
    adpbias_g_field = f"{asm['adpbias_gamma']:0>2x}"
    adpbias_b_field = f"{asm['adpbias_beta']:0>2x}"
    adpbias_i_field = f"{asm['adpbias_inp']:0>2x}"
    valid_field = '01'
    instr = '0x0' + adpbias_g_field + adpbias_b_field + 2*'0' + adpbias_i_field + \
             num_ts_field + num_v_field + mem_idx_field + 6*'0' + valid_field 
    return self.__produce_insn(addr, instr, 'W')

  def __gen_cfg_gb_ctrl(self, asm):
    # assembly: cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]
    # assumptions: all input types are integer
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00700010)
    num_timestep = f"{asm['num_ts']:0>4x}"
    num_v_field = f"{asm['num_v_o']:0>2x}" + f"{asm['num_v_i']:0>2x}"
    mem_id_field = f"{asm['mem_id_o']:0>2x}" + f"{asm['mem_id_i']:0>2x}"
    rnn_flag_field = f"{asm['is_rnn']:0>4x}"
    mode_field = f"{asm['mode']:0>2x}"
    valid_field = '01'
    instr = '0x0' + num_timestep + num_v_field + mem_id_field + rnn_flag_field + mode_field + valid_field

    return self.__produce_insn(addr, instr, 'W')
  
  def __gen_cfg_zeropadding(self, asm):
    # assembly: cfg_zeropad [mem_id], [num_v], [num_ts_1], [num_ts_2]
    # [mem_id]: the index of the memory to perform zeropadding
    # [num_v]: the number of the vectors in a timestep
    # [num_ts_1]: the staring index of the timestep to perform zeropadding
    # [num_ts_2]: the end index of the timestep to perform zeropadding
    # zeropadding are done from num_ts_1 to num_ts_2 - 1
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00A00010)
    num_timestep_field = f"{asm['num_ts_2']:0>4x}" + f"{asm['num_ts_1']:0>4x}"
    num_vector_field = f"{asm['num_v']:0>4x}"
    mem_idx_field = f"{asm['mem_id']:0>4x}"
    valid_field = '00000001'
    instr = '0x0' + num_timestep_field + num_vector_field + mem_idx_field + valid_field

    return self.__produce_insn(addr, instr, 'W')

  def __gen_cfg_attention(self, asm):
    # assembly: cfg_attention [mem_id_1], [mem_id_2], [num_v], [num_ts], [adpbias_1], [adpbias_2], [adpbias_3], [adpbias_4]
    # [mem_id_1]: memory index of encoder in large buffer
    # [mem_id_2]: memory index of decoder in large buffer
    # [num_v]: number of vectors of an encoder/decoder timestep
    # [num_ts]: number of encoder timesteps
    # [adpbias_1]: adpfloat bias for encoder output matrix
    # [adpbias_2]: adpfloat bias for input from encoder 
    # [adpbias_3]: adpfloat bias for softmax intermediate output 
    # [adpbias_4]: adpfloat bias for final attention output
    addr = hex(self.__FLEXNLP_BASE_ADDR + 0x00B00010)
    adpbias_field = f"{asm['adpbias_4']:0>2x}" + f"{asm['adpbias_3']:0>2x}"
    adpbias_field += f"{asm['adpbias_2']:0>2x}" + f"{asm['adpbias_1']:0>2x}"
    num_ts_field = f"{asm['num_ts']:0>8x}"
    num_v_field = f"{asm['num_v']:0>4x}"
    mem_idx_field = f"{asm['mem_id_2']:0>2x}" + f"{asm['mem_id_1']:0>2x}"
    valid_field = '00000001'
    instr = '0x' + adpbias_field + num_ts_field + num_v_field + mem_idx_field + valid_field
    
    return self.__produce_insn(addr, instr, 'W')

  # -------------------------------------------
  # function trigger instructions
  # -------------------------------------------
  def __gen_start(self, asm):
    # assembly: start [op]
    #   [op]: which op to trigger
    # assumption: 1: GBControl; 2: LayerReduce; 3: LayerNorm; 4: ZeroPadding; 5: Attention
    assert asm['op'] in (1,2,3,4,5), "unsupported op function trigger"
    addr = hex(self.__FLEXNLP_BASE_ADDR + asm['op'] * 0x10)
    return self.__produce_insn(addr, '0x01', 'W')




  