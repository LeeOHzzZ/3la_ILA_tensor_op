# ILA assembly for FlexNLP
This folder define the ILA tensor op assembly and ILA assembly instructions, and converter to flexnlp-ila program fragment

### Code generation flow
flexnlp-ila tensor assembly --> flexnlp-ila assembly --> flexnlp-ila program fragment --> flexnlp AXI commands

### Run the test flow
```bash
python3 test.py linear_layer [num_vector_in] [num_vector_out] [num_timestep] [is_bias]
python3 test.py lstm [num_vector_in] [num_vector_out] [num_timestep] [is_bias] [is_zero_first]
# example 1: python3 test.py linear_layer 16 4 10 1
#   [num_vector_out]: must be integer multiple of 4
# example 2: python3 test.py lstm 4 4 1 1 1
#   [is_zero_first]: If true, the initial cell state and hidden state would be zero
# emample 3: python3 attention_driver.py --num_ts 5 --num_v 4 --mem_idx_enc 0 --mem_idx_dec 0 
```
---

### FlexNLP-ILA tensor op assembly
- Load/Store
  - `store_act [timestep_idx], [idx]`
  - `store_wgt [wgt_idx]`
  - `store_bias [bias_idx]`
  - `store_wgt_i [wgt_idx]`
  - `store_wgt_h [wgt_idx]`
  - `store_bias_i [bias_idx]`
  - `store_bias_h [bias_idx]`
  - `store_beta [idx]`
  - `store_gamma [idx]`
  - `load_act [mem_idx], [idx]`
- Operation
  - `maxp [num_ts]`
  - `meanp [num_ts]`
  - `linear_layer [num_ts], [is_bias]`
  - `lstm_layer [num_ts], [is_bias], [is_zero_first]`
  - `layernorm [num_ts]`

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
  - `cfg_ly_norm [mem_idx], [num_v], [num_ts], [adpbias_inp], [adpbias_beta], [adpbias_gamma]`
  - `cfg_gb_ctrl [mode], [is_rnn], [mem_id_i], [mem_id_o], [num_v_i], [num_v_o], [num_ts]`
  - `cfg_zeropad [mem_id], [num_v], [num_ts_1], [num_ts_2]`
  - `cfg_attention [mem_id_1], [mem_id_2], [num_v], [num_ts], [adpbias_1], [adpbias_2], [adpbias_3], [adpbias_4]`
  - `start [op]`
- Load/Store
  - `write_v [vector_name], [addr]`
  - `read_v [vector_name], [addr]`
