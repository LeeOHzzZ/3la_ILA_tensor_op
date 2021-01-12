"""
This file contains parameters for vta-ila
"""
VTA_INSTR_BITWIDTH = 128
# instruction width
VTA_OPCODE_WIDTH = 3

# =====================
# memory instruction
# =====================
# /*! Memory type field bitwidth */
VTA_MEMOP_ID_BITWIDTH = 2
# /*! Load/Store Instruction: DRAM address width*/
VTA_MEMOP_SRAM_ADDR_BITWIDTH = 16
# /*! Load/Store Instruction: DRAM address width*/
VTA_MEMOP_DRAM_ADDR_BITWIDTH = 32
# /*! Load/Store Instruction: transfer size width*/
VTA_MEMOP_SIZE_BITWIDTH = 16
# /*! Load/Store Instruction: stride size width*/
VTA_MEMOP_STRIDE_BITWIDTH = 16
# /*! Load/Store Instruction: padding width*/
VTA_MEMOP_PAD_BITWIDTH = 4
# /*! Load/Store Instruction: padding value encoding width*/
VTA_MEMOP_PAD_VAL_BITWIDTH = 2

# /*! Mem ID constant: uop memory */
VTA_MEM_ID_UOP = 0
# /*! Mem ID constant: weight memory */
VTA_MEM_ID_WGT = 1
# /*! Mem ID constant: input memory */
VTA_MEM_ID_INP = 2
# /*! Mem ID constant: accumulator/bias memory */
VTA_MEM_ID_ACC = 3
# /*! Mem ID constant: output store buffer */
VTA_MEM_ID_OUT = 4

# opcode type
VTA_OPCODE_LOAD = 0
VTA_OPCODE_STORE = 1
VTA_OPCODE_GEMM = 2
VTA_OPCODE_ALU = 4