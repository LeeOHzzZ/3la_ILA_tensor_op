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

**Work In Progress**
