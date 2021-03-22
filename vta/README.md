# ILA assembly for VTA
This folder defines the ILA asssemly for vta-ila instructions, and contains a converter to vta-ila program fragment

## Building the converter
The converter are implemented in Python.  
The converter takes two inputs, the assembly JSON file and data_mapping JSON file  
Two testcases are provided.  
- simple assembly sample, you can simply run it by:  
  ``` bash
    ./run_simple_test.sh
  ```
- Blocked GEMM testcase, you can simply run it by:
  ``` bash
    ./run_gemm.sh
  ```
## VTA-ILA assembly documentation
VTA-ILA assembly are similar to the VTA CISC instructions. A short introduction can be found [here](https://tvm.apache.org/docs/vta/dev/hardware.html#instruction-set-architecture).  
Because VTA-ILA didn't model dependency between different FIFOs, dependency flag variables are not included in VTA-ILA assembly.  
VTA is an accelerator targeting at tensors, and the smallest data unit is tensor, which has a fixed dimension once VTA's configuration is set. More information can be found [here](https://tvm.apache.org/docs/vta/dev/hardware.html#compute-module)  

### Load
VTA-ILA assembly has four different `load` instructions: `load_bias`, `load_wgt`, `load_inp` and `load_uop`.  
VTA-ILA, modeling the HLS implementation of VTA [here](https://github.com/apache/tvm-vta/tree/main/hardware/xilinx/src) (#87ce9ac), uses the same notion for `sram_id` and `dram_id`. **These two indexes are not exact addresses in memory, but indexes to the source buffer and destination buffer**.

For example, vta-ila has a `virtual_bias_buffer` for holding bias data in a "vitural DRAM", a `acc_buffer`, modeling part of the SRAM in VTA, and each bias are 4 byte. Then `sram_id = 1` is the index for the second bias data in the `acc_buffer`, its physical memory address would be `base_addr_of(bias_buffer) + 4`. But we don't care about the physical memory address in the VTA-ILA model.

- `load_bias [sram_id], [dram_id], [y_size], [x_size], [x_stride]`  
VTA load bias tensor from external DRAM to its internal SRAM (`acc_buffer`)
  - `sram_id`: index of the starting entry of the destination buffer in SRAM to load data to
  - `dram_id`: index of the starting entry of the source buffer in DRAM to load data from
  - `y_size`: number of tensor in y dimension of the whole matrix to be loaded
  - `x_size`: number of tensor in x dimension of the whole matrix to be loaded
  - `x_stride`: number of tensor for stride in x dimension when loading  

  ```assembly
  # example
    load_bias 1, 2, 5, 6, 1
  # load 5*6=30 tensors of bias with x_stride=1, starting from virtual_bias_buffer[2]
  # and store them to VTA starting from acc_buffer[1]
  ```
***
- `load_wgt [sram_id], [dram_id], [y_szie], [x_size], [x_stride]`
VTA load weight tensors from external DRAM to its interal SRAM (`wgt_buffer`)
  - `sram_id`: index of the starting entry of the destination buffer in SRAM to load data to
  - `dram_id`: index of the starting entry of the source buffer in DRAM to load data from
  - `y_size`: number of tensor in y dimension of the whole matrix to be loaded
  - `x_size`: number of tensor in x dimension of the whole matrix to be loaded
  - `x_stride`: number of tensor for stride in x dimension when loading   
***
- `load_inp [sram_id], [dram_id], [y_size], [x_size], [x_stride], [y_pad0], [y_pad1], [x_pad0], [x_pad1]`  
VTA load input tensors from external DRAM to its interal SRAM (`inp_buffer`)
  - `sram_id`: index of the starting entry of the destination buffer in SRAM to load data to
  - `dram_id`: index of the starting entry of the source buffer in DRAM to load data from
  - `y_size`: number of tensor in y dimension of the whole matrix to be loaded
  - `x_size`: number of tensor in x dimension of the whole matrix to be loaded
  - `x_stride`: number of tensor for stride in x dimension when loading
  - `y_pad0`: padding before x dimension tensors
  - `y_pad1`: padding after x dimension tensors
  - `x_pad0`: padding before tensors in x dimension
  - `x_pad1`: padding after tensors in x dimension
***
- `load_uop [sram_id], [dram_id], [x_size]`
VTA load micro-op from external DRAM to its internal SRAM (`uop_buffer`)
  - `sram_id`: index of the starting entry of the destination buffer in SRAM to load data to
  - `dram_id`: index of the starting entry of the source buffer in DRAM to load data from
  - `x_size`: number of micro-ops to be loaded



### Store
Only one store instruction is supported in VTA, which is store the accumulation results from VTA's SRAM back to DRAM. In VTA-ILA, `store_acc` will store results to the `virtual_output_buffer`
- `store_acc [sram_id], [dram_id], [y_size], [x_size], [x_stride]`
  - `sram_id`: index of the starting entry of the source buffer in SRAM to store data from
  - `dram_id`: index of the starting entry of the destination buffer in DRAM to store data to
  - `y_size`: number of tensor in y dimension of the whole matrix to be stored
  - `x_size`: number of tensor in x dimension of the whole matrix to be stored
  - `x_stride`: number of tensor for stride in x dimension when storing

### GEMM
[A short introduction on VTA GEMM instruction](https://tvm.apache.org/docs/vta/dev/hardware.html#compute-module)  
VTA GEMM will perform matrix-matrix/vector multiplication with inputs from `input_buffer` and weights from `weight_buffer` and accumulate the results in `acc_buffer`.  
- `gemm [reset_f], [uop_bgn], [uop_end], [iter_o], [iter_i], [dst_fo], [dst_fi], [src_fo], [src_fi], [wgt_fo], [wgt_fi]`
  - `reset_f`: reset accumulator after multiplication
  - `uop_bgn`: begining idx of micro_op in the uop buffer for gemm
  - `uop_end`: ending idx of micro_op in the uop buffer for gemm
  - `iter_o`: Outer loop iterations
  - `iter_i`: Inner loop iterations
  - `dst_fo`: Outer loop accumulator (bias) tensor index increment factor
  - `dst_fi`: Inner loop accumulator (bias) index increment factor
  - `src_fo`: Outer loop input tensor index increment factor
  - `src_fi`: Inner loop input tensor index increment factor
  - `wgt_fo`: Outer loop weight tensor index increment factor
  - `wgt_fi`: Inner loop weight tensor index increment factor

### ALU
VTA ALU will perform vector level alu instructions including: min, max, shr, add
- `alu [reset_f], [uop_bgn], [uop_end], [iter_o], [iter_i], [dst_fo], [dst_fi], [src_fo], [src_fi], [alu_op], [use_imm], [imm]
  - `reset_f`: reset accumulator after multiplication
  - `uop_bgn`: begining idx of micro_op in the uop buffer for gemm
  - `uop_end`: ending idx of micro_op in the uop buffer for gemm
  - `iter_o`: Outer loop iterations
  - `iter_i`: Inner loop iterations
  - `dst_fo`: Outer loop accumulator (bias) tensor index increment factor
  - `dst_fi`: Inner loop accumulator (bias) index increment factor
  - `src_fo`: Outer loop input tensor index increment factor
  - `src_fi`: Inner loop input tensor index increment factor
  - `alu_op`: 0 - min; 1 - max; 2 - add; 3 - shr
  - `use_imm`: whether 2nd op of the alu instruction is immediate
  - `imm`: immediate value
