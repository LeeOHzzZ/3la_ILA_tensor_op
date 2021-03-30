# ILA assembly for HLSCNN
This folder contains the specification of ILA assembly instructions for HLSCNN and converter to hlscnn-ila program fragment and HLSCNN AXI commands

### HLSCNN-ILA assembly
- `SpadWr [addr]`
  - `[addr]`: hex string of SPAD address to write
  - SPAD0 address: 0x04000 ~ 0x24000 (20KB)
  - SPAD1 address: 0x24000 ~ 0x44000
  - when write to a SPAD address, HLSCNN will get the data from SoC memory through its AXI master port instead of using the data from the AXI slave port ==> no data needed for this instruction

- `SpadRd [addr]`
  - `[addr]`: hex string of SPAD address to read
  - SPAD0 address: 0x04000 ~ 0x24000 
  - SPAD1 address: 0x24000 ~ 0x44000

- `VirMemWr [addr], [data]`
  - `[addr]`: hex string of virtual SoC memory address to write into
  - `[data]`: hex string of 128bit data
  - The Virtual SOC memory in ILA has an offset address of 0x50000.
    - In order to write to VirMem, the address field should be greater than 0x50000
    - Thus, when setting ActBase as 0x10000, the vir_mem_wr for act
      should start from 0x10000 + 0x50000 = 0x60000

- `CfgSoCMemBaseAddr [base_addr]`
  - `[base_addr]`: hex string of base address for reading from SoC Memory
  - this value is set for reading the data from SoC memory through AXI master interface

- `CfgSocMemRdWrLen [length]`
  - `[length]`: int of read/write operation length of HLSCNN through AXI master port
  - This value should be set to 1 for now. HLSCNN may not support more than 1 correctly.

- `CfgConvActBaseAddr [base_addr]`
  - `[base_addr]`: hex string, the base address of the activation in the SoC memory for convolution

- `CfgConvWgtBaseAddr [base_addr]`
  - `[base_addr]`: hex string of the base address for weights in the SPAD0 for convolution

- `CfgConvOutBaseAddr [base_addr]`
  - `[base_addr]`: hex_string of base address of outputs in SPAD1 for convolution

- `CfgConvInpSize [inp_cols], [inp_rows], [inp_chans]`
  - `[inp_cols]`: int of input columns of convolution
  - `[inp_rows]`: int of input rows of convolution
  - `[inp_chans]`: int of input channels of convolution

- `CfgConvOutSize [out_cols], [out_rows], [out_chans]`
  - `[out_cols]`: int of output tensor columns after convolution
  - `[out_rows]`: int of output tensor rows after convolution
  - `[out_chans]`: int of output tensor chanels after convolution

- `CfgConvKernelSize [kernel_cols], [kernel_rows], [kernel_c_stride], [kernel_r_stride]`
  - `[kernel_cols]`: int of kernel columns of the convolution
  - `[kernel_rows]`: int of kernel rows of the convolution
  - `[kernel_c_stride]`: int of kernel column stride
  - `[kernel_r_stride]`: int of kernel rows stride

- `CfgConvChan [chan_bias], [is_bias], [is_relu], [is_accum], [kernel_num], [is_wb]`
  - `[chan_bias]`: int of equivalent raw bits of 16-bit fixed-point chanel bias value
  - `[is_bias]`: int of wheter enable bias on convolution
  - `[is_relu]`: int of whether enable relu on convolution
  - `[is_accum]`: int of whether enable accumulation in the convolution
  - `[kernel_num]`: int of number of kernels/output channels of the convolution
  - `[is_wb]`: int of an unclear operation. Can set it to 0 for ILA simulation

- `ConvStart`
  - trigger the convolution calculation
