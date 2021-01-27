# ILA assembly for FlexNLP
This folder define the ILA tensor op assembly and ILA assembly instructions, and converter to flexnlp-ila program fragment

### Code generation flow
flexnlp-ila tensor assembly --> flexnlp-ila assembly --> flexnlp-ila program fragment --> flexnlp AXI commands

### Run the test flow
```bash
python3 linear_layer_testflow.py [num_vector_in] [num_vector_out] [num_timestep] [is_bias]
# example: python3 linear_layer_testflow.py 16 4 10 1
# [num_vector_out] must be integer multiple of 4
```
---

### FlexNLP-ILA tensor op assembly
- Load/Store
  - `store_act [timestep_idx], [idx]`
  - `store_wgt [wgt_idx]`
  - `store_bias [bias_idx]`
- Operation
  - `maxp [num_ts]`
  - `linear_layer [num_ts], [is_bias]`

---

### FlexNLP-ILA assembly
- PE instructions
  - `pe_cfg_rnn_layer_sizing [pe_idx], [is_zero], [is_cluster], [is_bias], [num_mngr], [num_v_out]`
  - `pe_cfg_mngr [pe_idx], [mngr_idx], [is_zero], [adpbias_wgt], [adpbias_bias], [adpbias_inp], [num_v_in], [base_wgt], [base_bias], [base_inp]`
  - `pe_cfg_act_mngr [pe_idx], [is_zero], [adpfloat_bias], [num_insn], [num_v_out], [buf_base], [out_base]`
  - `pe_cfg_act_v [pe_idx], [v_idx], [insn_0], ..., [insn_15]`
- GB instructions
  - `cfg_mmngr_gb_large [base_0], [num_v_0] (, [base_1], [num_v_1], ..., [base_3], [num_v_3])`
  - `cfg_mmgnr_gb_small [base_0] (, [base_1], ..., [base_7])`
  - `cfg_ly_reduce [mode], [mem_idx], [num_v], [num_ts]`
  - `cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]`
  - `start [op]`
- Load/Store
  - `write_v [vector_name], [addr]`
  - `read_v [vector_name], [addr]`
