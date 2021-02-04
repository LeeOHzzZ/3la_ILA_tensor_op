class ts_asm_converter:
  
  __FLEXNLP_VECTOR_SIZE = 16
  __FLEXNLP_GBCORE_NUM_BANKS = 16
  __FLEXNLP_GB_LARGE_BUF_BASE = '0x33500000'
  __GB_CONTROL_START = 1
  __GB_LAYER_REDUCE_START = 2

  __gb_large_buf_mem_base = [0, 0, 0, 0]

  def __init__(self, ts_asm_list, data_lib):
    self.ts_asm_list = ts_asm_list
    self.data_lib = data_lib

  def to_ila_asm(self):
    self.asm_list = []
    for asm in self.ts_asm_list:
      self.asm_list += self.generate_ila_asm(asm)
    return self.asm_list
  
  # ----------------------------------------------------
  # ----------------------------------------------------

  def get_gb_large_addr_offset(self, ts_idx, num_vector, vector_idx):
    # calculate the start address offset in the gbcore large buffer
    timestep_size = num_vector * self.__FLEXNLP_VECTOR_SIZE
    group_size = timestep_size * self.__FLEXNLP_GBCORE_NUM_BANKS
    gb_buf_row_size = self.__FLEXNLP_GBCORE_NUM_BANKS * self.__FLEXNLP_VECTOR_SIZE
    out = (int(ts_idx/self.__FLEXNLP_GBCORE_NUM_BANKS)) * group_size + \
          (ts_idx%self.__FLEXNLP_GBCORE_NUM_BANKS) * self.__FLEXNLP_VECTOR_SIZE + \
          gb_buf_row_size * vector_idx
    return out

  def get_gb_large_abs_addr(self, mem_idx, ts_idx, num_v, v_idx):
    out = self.get_gb_large_addr_offset(ts_idx, num_v, v_idx)
    out += int(self.__FLEXNLP_GB_LARGE_BUF_BASE, base=16)
    out += self.__gb_large_buf_mem_base[mem_idx]
    return out

  def get_pe_hidden_wgt_offset(self, num_v_in, num_v_out):
    # return the offset address for hidden weight matrix in PE
    # this address is vector level address
    return num_v_in * num_v_out * 16

  def get_pe_base_bias_v_1(self, num_v):
    # get bias base address in pe input buffer (vector level address)
    return num_v
  
  def get_pe_base_bias_v_2(self, num_v_in, num_v_out):
    # get bias base address in pe input buffer for hidden state (vector level addr)
    return self.get_pe_base_h_input(num_v_in, num_v_out) + num_v_out
  
  def get_pe_base_h_input(self, num_v_in, num_v_out):
    # get base address for hidden state input in the PE input buffer (vector level addr)
    return self.get_pe_base_bias_v_1(num_v_in) + num_v_out
    
  def get_gb_base_addr_1(self, num_ts, num_v_in):
    # get base address for gb large buffer of memory index 1 (vector_level)
    return int(num_ts/16 + 2) * 16 * num_v_in  
  
  def gen_store_bias_helper(self, bias_idx, base_addr, num_v_bias):
    ret = []
    for pe_idx in range(4):
      for v in range(num_v_bias):
        addr = 0x34600000 + pe_idx * 0x01000000 + base_addr + v*16
        bias_v_idx = pe_idx * num_v_bias + v
        v_name = '{}.{}'.format(bias_idx, bias_v_idx)
        ret.append({
          'name' : 'write_v',
          'vector_name' : v_name,
          'addr' : hex(addr)
        })
    return ret
  # ==================================================================
  # ================================================================== 
  def generate_ila_asm(self, asm):
    """
    generate flexnlp-ila assembly from ILA tensor assembly
    """
    asm_types = ['store_act', 'store_wgt', 'store_bias', 'load_act']
    asm_types += ['store_wgt_i', 'store_wgt_h', 'store_bias_i', 'store_bias_h']
    asm_types += ['maxp', 'linear_layer', 'lstm_layer']
    # this instruction added for simulation
    asm_types += ['wait_irq']
    assert asm['name'] in asm_types, "{} is a not supported ILA assembly".format(asm['name'])

    # asm format: asm_name arg_0 [, arg_1 ...]
    if asm['name'] == "store_act":
      return self.gen_store_act(asm)
    if asm['name'] == 'store_wgt':
      return self.gen_store_wgt(asm)
    if asm['name'] == 'store_bias':
      return self.gen_store_bias(asm)
    if asm['name'] == 'load_act':
      return self.gen_load_act(asm)

    if asm['name'] == 'store_wgt_i':
      return self.gen_store_wgt_i(asm)
    if asm['name'] == 'store_wgt_h':
      return self.gen_store_wgt_h(asm)
    if asm['name'] == 'store_bias_i':
      return self.gen_store_bias_i(asm)
    if asm['name'] == 'store_bias_h':
      return self.gen_store_bias_h(asm)

    if asm['name'] == "maxp":
      return self.gen_maxp(asm)
    if asm['name'] == 'linear_layer':
      return self.gen_linear_layer(asm)
    if asm['name'] == 'lstm_layer':
      return self.gen_lstm_layer(asm)

    if asm['name'] == 'wait_irq':
      return [{'name' : 'wait_irq'}]

  # ------------------------------------------
  # Load/Store instructions
  # ------------------------------------------
  def gen_store_act(self, asm):
    # format: store_act [timestep_idx], [idx]
    # timestep_idx: index of the tensor place holder
    # idx: timestep index in the flexnlp GB large buffer
    # description: 
    #   store timestep input data into FlexNLP GB large buffer
    num_vector_in = self.data_lib['gb_num_vector_in']
    tensor_idx = asm['timestep_idx']
    idx = asm['idx']
    ret = []

    for v in range(num_vector_in):
      v_name = tensor_idx + '.' + str(v)
      addr = hex(self.get_gb_large_abs_addr(0, idx, num_vector_in, v))
      ret.append({
        'name' : 'write_v',
        'vector_name' : v_name,
        'addr' : addr
      })
    return ret

  def gen_store_wgt(self, asm, addr_offset = 0):
    # format: store_wgt [wgt_idx]
    #   [weight_idx]: weight matrix symbol
    # description: 
    #   This instruction would store the weight matrix data into
    #   FlexNLP four PEs, according to FlexNLP tiling convention
    # assumptions:
    #   data_lib already has weight matrix dimensions in tiles (16x16)
    wgt_idx = asm['wgt_idx']
    num_tiles = self.data_lib[wgt_idx + '_num_tile']
    ret = []
    for t in range(num_tiles):
      pe_idx = int(t/(num_tiles/4))
      pe_base_addr = 0x34500000 + pe_idx*0x01000000
      tile_base_addr = pe_base_addr + addr_offset + (t%int(num_tiles/4))*16*16
      for v in range(16):
        v_name = '{}.t{}.{}'.format(wgt_idx, t, v)
        addr = tile_base_addr + v*16
        ret.append({
          'name' : 'write_v',
          'vector_name' : v_name,
          'addr' : hex(addr)
        })
    return ret

  def gen_store_bias(self, asm):
    # format: store_bias [bias_idx]
    # [bias_idx]: string, bias vector symbol
    # description:
    #   Store bias into PE's input buffer, divided into 4 segments
    # assumption:
    #   data_lib already provide num_v_out;
    #   need to leave space for timestep data
    bias_idx = asm['bias_idx']
    # num_v_in are the same for gb and pe
    num_v_in = self.data_lib['gb_num_vector_in']
    # num_v_out are different for gb and pe
    num_v_out_gb = self.data_lib['gb_num_vector_out']
    num_v_out_pe = num_v_out_gb >> 2
    base_bias = self.get_pe_base_bias_v_1(num_v_in) * 16 # byte level address
    return self.gen_store_bias_helper(bias_idx, base_bias, num_v_out_pe)

  def gen_load_act(self, asm):
    # format: load_act [mem_idx], [ts_idx]
    # [mem_idx]: int, memory_idx in the FlexNLP large buffer
    # [ts_idx]: int, timestep index to be loaded
    # description:
    #   load activations from flexnlp gb_large_buffer
    # assumption:
    #   hard to implement return tensor symbol
    num_v_out = self.data_lib['gb_num_vector_out']
    ret = []
    for v in range(num_v_out):
      addr = self.get_gb_large_abs_addr(asm['mem_idx'], asm['ts_idx'], num_v_out, v)
      ret.append({
        'name' : 'read_v',
        'addr' : hex(addr)
      })
    return ret

  def gen_store_wgt_i(self, asm):
    # format: store_wgt_i [wgt_idx]
    #   [wgt_idx]: string, weight matrix symbol
    # description:
    #   Store input weights for RNN operation
    return self.gen_store_wgt(asm)
  
  def gen_store_wgt_h(self, asm):
    # format: store_wgt_h [wgt_idx]
    #   [wgt_idx]: string, weight matrix symbol
    # description:
    #   store weight for hidden states for RNN operation
    num_v_in = self.data_lib['gb_num_vector_in']
    num_v_out = self.data_lib['gb_num_vector_out']
    addr_offset = 16 * self.get_pe_hidden_wgt_offset(num_v_in, num_v_out)
    return self.gen_store_wgt(asm, addr_offset = addr_offset)
  
  def gen_store_bias_i(self, asm):
    # format: store_bias_i [bias_idx]
    #   [bias_idx]: bias vector symbol
    # description:
    #   store input bias into PE for RNN operation
    bias_idx = asm['bias_idx']
    num_v_in = self.data_lib['gb_num_vector_in']
    num_v_out_gb = self.data_lib['gb_num_vector_out']
    # num_v_out_pe = int(num_v_out_gb/4)
    num_v_bias_pe = num_v_out_gb
    # use num_v_out_gb for this function
    # pe would store the whole input timestep into PE input buffer
    base_bias = self.get_pe_base_bias_v_1(num_v_in) * 16 # byte level address
    return self.gen_store_bias_helper(bias_idx, base_bias, num_v_bias_pe)
  
  def gen_store_bias_h(self, asm):
    # format: store_bias_h [bias_idx]
    #   [bias_idx]: bias vector symbol
    # description:
    #   store bias for hidden states into PE for RNN operation
    bias_idx = asm['bias_idx']
    num_v_in = self.data_lib['gb_num_vector_in']
    num_v_out_gb = self.data_lib['gb_num_vector_out']
    num_v_bias_pe = num_v_out_gb
    # use num_v_out_gb for this function
    # pe would store the whole input timestep into PE input buffer
    base_bias = self.get_pe_base_bias_v_2(num_v_in, num_v_out_gb) * 16 # byte level address
    return self.gen_store_bias_helper(bias_idx, base_bias, num_v_bias_pe)

  # ------------------------------------------
  # op assembly
  # ------------------------------------------
  def gen_maxp(self, asm):
    # format: maxp [num_ts]
    # [num_ts]: number of timesteps to maxpooled in the FlexNLP GB large buffer
    # Assumption: 
    #   timestep dimension is given in the data_lib file
    ret = []
    # set up gb memory manager
    ret.append({
      'name' : 'cfg_mmngr_gb_large',
      'base_0' : hex(0x0),
      'num_v_0' : self.data_lib['gb_num_vector_in']
    })
    # set up gb layer reduce configuration
    ret.append({
      'name' : 'cfg_ly_reduce',
      'mode' : 0,
      'mem_idx' : 0,
      'num_v' : self.data_lib['gb_num_vector_in'],
      'num_ts' : asm['num_ts']
    })
    # trigger layer reduce start
    ret.append({
      'name' : 'start',
      'op' : self.__GB_LAYER_REDUCE_START
    })
    return ret

  def gen_linear_layer(self, asm):
    # format: linear_layer [num_ts], [is_bias]
    # [num_ts]: number of input timesteps for linear layer
    # [is_bias]: whether apply bias to linear layer
    # assumptions: 
    #   1. linear layer need to set two memory indexes (num_v_out is different from num_v_in),
    #      Thus, I assume [num_ts] is equal to all the timesteps previously stored in gb. The 
    #      rest may be overwritten.
    ret = []
    gb_num_v_in = self.data_lib['gb_num_vector_in']
    gb_num_v_out = self.data_lib['gb_num_vector_out']
    pe_num_v_out = int(gb_num_v_out/4)

    # set up PE related assembly
    for pe_idx in range(4):
      # set up pe_cfg_layer_sizing
      # pe_cfg_rnn_layer_sizing [pe_idx], [is_zero], [is_cluster], [is_bias], [num_mngr], [num_v_out]
      ret.append({
        'name' : 'pe_cfg_rnn_layer_sizing',
        'pe_idx' : pe_idx,
        'is_zero' : 0,
        'is_cluster' : 0,
        'is_bias' : asm['is_bias'],
        'num_mngr' : 1,
        'num_v_out' : pe_num_v_out
      })
      # set up pe_cfg_mngr
      # only need the first memory manager for linear layer
      # pe_cfg_mngr [pe_idx], [mngr_idx], [is_zero], [adpbias_wgt], [adpbias_bias], \
      # [adpbias_inp], [num_v_in], [base_wgt], [base_bias], [base_inp]
      ret.append({
        'name' : 'pe_cfg_mngr',
        'pe_idx' : pe_idx,
        'mngr_idx' : 1,
        'is_zero' : 0,
        'adpbias_wgt' : self.data_lib['adpbias_wgt'],
        'adpbias_bias' : self.data_lib['adpbias_bias'],
        'adpbias_inp' : self.data_lib['adpbias_inp'],
        'num_v_in' : gb_num_v_in,
        'base_wgt' : 0,
        'base_bias' : self.get_pe_base_bias_v_1(gb_num_v_in),
        'base_inp' : 0
      })
      # set up pe_cfg_act_mngr
      # pe_cfg_act_mngr [pe_idx], [is_zero], [adpfloat_bias], [num_insn], [num_v_out], [buf_base], [out_base]
      ret.append({
        'name' : 'pe_cfg_act_mngr',
        'pe_idx' : pe_idx,
        'is_zero' : 0,
        'adpfloat_bias' : self.data_lib['adpbias_pe_act'],
        'num_insn' : 2,
        'num_v_out' : pe_num_v_out,
        'buf_base' : 0,
        'out_base' : pe_idx * pe_num_v_out
      })
      # set up pe_config_act_v: micro instructions for act
      # this is fixed for linear layer
      # pe_cfg_act_v [pe_idx], [v_idx], [insn_0], ..., [insn_15]
      ret.append({
        'name' : 'pe_cfg_act_v',
        'pe_idx' : pe_idx,
        'v_idx' : 1,
        'insn_0' : '0x30',
        'insn_1' : '0x40'
      })
    
    # set up gb memory manager
    num_ts = asm['num_ts']
    # this value is vector level
    base_addr_1 = self.get_gb_base_addr_1(num_ts, gb_num_v_in)
    # temporarily set this index as global variable.
    # should come up with better solution
    self.__gb_large_buf_mem_base[1] = base_addr_1 * 16

    # set up gb large memory manager
    ret.append({
      'name' : 'cfg_mmngr_gb_large',
      'base_0' : hex(0),
      'num_v_0' : gb_num_v_in,
      'base_1' : hex(base_addr_1),
      'num_v_1' : gb_num_v_out
    })
    # set up gb control configure
    # cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]
    ret.append({
      'name' : 'cfg_gb_ctrl',
      'mode' : 0,
      'is_rnn' : 0,
      'mem_id_i' : 0,
      'mem_id_o' : 1,
      'num_v_i' : self.data_lib['gb_num_vector_in'],
      'num_v_o' : self.data_lib['gb_num_vector_out'],
      'num_ts' : num_ts
    })
    # trigger start
    ret.append({
      'name' : 'start',
      'op' : self.__GB_CONTROL_START
    })

    return ret 

  def gen_lstm_layer(self, asm):
    # format: lstm_layer [num_ts], [is_bias], [is_zero_first]
    #   [num_ts]: number of timestep for lstm_layer
    #   [is_bias]: bias to lsm
    #   [is_zero]: set initial hidden state, cell state as zero
    ret = []
    gb_num_v_in = self.data_lib['gb_num_vector_in']
    gb_num_v_out = self.data_lib['gb_num_vector_out']
    # this output is after activations, not after pe_core
    pe_num_v_out = gb_num_v_out >> 2

    # set up PE related assembly
    for pe_idx in range(4):
      # set up pe_cfg_layer_sizing
      # pe_cfg_rnn_layer_sizing [pe_idx], [is_zero], [is_cluster], [is_bias], [num_mngr], [num_v_out]
      ret.append({
        'name' : 'pe_cfg_rnn_layer_sizing',
        'pe_idx' : pe_idx,
        'is_zero' : asm['is_zero_first'],
        'is_cluster' : 0,
        'is_bias' : asm['is_bias'],
        'num_mngr' : 2,
        'num_v_out' : pe_num_v_out*4
      })
      # set up pe_cfg_mngr
      # pe_cfg_mngr [pe_idx], [mngr_idx], [is_zero], [adpbias_wgt], [adpbias_bias], \
      # [adpbias_inp], [num_v_in], [base_wgt], [base_bias], [base_inp]
      # 1st one is for input states
      ret.append({
        'name' : 'pe_cfg_mngr',
        'pe_idx' : pe_idx,
        'mngr_idx' : 1,
        'is_zero' : 0,
        'adpbias_wgt' : self.data_lib['adpbias_wgt'],
        'adpbias_bias' : self.data_lib['adpbias_bias'],
        'adpbias_inp' : self.data_lib['adpbias_inp'],
        'num_v_in' : gb_num_v_in,
        'base_wgt' : 0x0,
        'base_bias' : self.get_pe_base_bias_v_1(gb_num_v_in),
        'base_inp' : 0x0
      })
      # 2nd one is for hidden states
      ret.append({
        'name' : 'pe_cfg_mngr',
        'pe_idx' : pe_idx,
        'mngr_idx' : 2,
        'is_zero' : asm['is_zero_first'],
        'adpbias_wgt' : self.data_lib['adpbias_wgt'],
        'adpbias_bias' : self.data_lib['adpbias_bias'],
        'adpbias_inp' : self.data_lib['adpbias_inp'],
        'num_v_in' : gb_num_v_out,
        'base_wgt' : self.get_pe_hidden_wgt_offset(gb_num_v_in, gb_num_v_out),
        'base_bias' : self.get_pe_base_bias_v_2(gb_num_v_in, gb_num_v_out),
        'base_inp' : self.get_pe_base_h_input(gb_num_v_in, gb_num_v_out)
      })
      # set up pe_cfg_act_mngr
      # pe_cfg_act_mngr [pe_idx], [is_zero], [adpfloat_bias], [num_insn], [num_v_out], [buf_base], [out_base]
      ret.append({
        'name' : 'pe_cfg_act_mngr',
        'pe_idx' : pe_idx,
        'is_zero' : asm['is_zero_first'],
        'adpfloat_bias' : self.data_lib['adpbias_pe_act'],
        'num_insn' : 0x18,
        'num_v_out' : pe_num_v_out,
        'buf_base' : 0,
        'out_base' : pe_idx * pe_num_v_out
      })
      # set up pe_config_v: micro instructions for pe_act
      # this is fixed for a standard LSTM
      # pe_cfg_act_v [pe_idx], [v_idx], [insn_0], ..., [insn_15]
      # 1st vector instruction table
      ret.append({
        'name' : 'pe_cfg_act_v',
        'pe_idx' : pe_idx,
        'v_idx' : 1,
        'insn_0' : '0x30',
        'insn_1' : '0x34',
        'insn_2' : '0x81',
        'insn_3' : '0xa0',
        'insn_4' : '0x34',
        'insn_5' : '0x38',
        'insn_6' : '0x86',
        'insn_7' : '0xb4',
        'insn_8' : '0x91',
        'insn_9' : '0x34',
        'insn_10' : '0x38',
        'insn_11' : '0x86',
        'insn_12' : '0xa4',
        'insn_13' : '0x18',
        'insn_14' : '0x96',
        'insn_15' : '0x81'
      })
      ret.append({
        'name' : 'pe_cfg_act_v',
        'pe_idx' : pe_idx,
        'v_idx' : 2,
        'insn_0' : '0x20',
        'insn_1' : '0xb0',
        'insn_2' : '0x34',
        'insn_3' : '0x38',
        'insn_4' : '0x86',
        'insn_5' : '0xa4',
        'insn_6' : '0x91',
        'insn_7' : '0x40'
      })
    
    # set up gb related assembly
    # set up gb_memory manager
    num_ts = asm['num_ts']
    # TODO: now set the output result to memory index 1 as default
    base_addr_1 = self.get_gb_base_addr_1(num_ts, gb_num_v_in)
    self.__gb_large_buf_mem_base[1] = base_addr_1 << 4

    # set up gb_memory_manger_large
    # cfg_mmngr_gb_large [base_0], [num_v_0] (, [base_1], [num_v_1], ..., [base_3], [num_v_3])
    ret.append({
      'name' : 'cfg_mmngr_gb_large',
      'base_0' : hex(0x0),
      'num_v_0' : gb_num_v_in,
      'base_1' : hex(base_addr_1),
      'num_v_1' : gb_num_v_out
    })

    # set up gb control configuration
    # cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]
    ret.append({
      'name' : 'cfg_gb_ctrl',
      'mode' : 0,
      'is_rnn' : 1,
      'mem_id_i' : 0,
      'mem_id_o' : 1,
      'num_v_i' : gb_num_v_in,
      'num_v_o' : gb_num_v_out,
      'num_ts' : num_ts
    })

    # trigger start
    ret.append({
      'name' : 'start',
      'op' : self.__GB_CONTROL_START
    })

    return ret

