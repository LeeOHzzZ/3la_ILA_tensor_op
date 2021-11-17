"""
This file contains parameters for vta-ila
"""
VTA_INSTR_BITWIDTH = 128
# instruction width
VTA_OPCODE_BITWIDTH = 3

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

# ====================
# GEMM
# ====================
VTA_GEMM_UOP_BEGIN_BITWIDTH = 13
VTA_GEMM_UOP_END_BITWIDTH = 14
VTA_GEMM_ITER_OUT_BITWIDTH = 14
VTA_GEMM_ITER_IN_BITWIDTH = 14
VTA_GEMM_DST_FACTOR_OUT_BITWIDTH = 11
VTA_GEMM_DST_FACTOR_IN_BITWIDTH = 11
VTA_GEMM_SRC_FACTOR_OUT_BITWIDTH = 11
VTA_GEMM_SRC_FACTOR_IN_BITWIDTH = 11
VTA_GEMM_WGT_FACTOR_OUT_BITWIDTH = 10
VTA_GEMM_WGT_FACTOR_IN_BITWIDTH = 10

# ====================
# ALU
# ====================
VTA_ALU_RESET_FLAG_BITWIDTH = 1
VTA_ALU_UOP_BEGIN_BITWIDTH = 13
VTA_ALU_UOP_END_BITWIDTH = 14
VTA_ALU_ITER_OUT_BITWIDTH = 14
VTA_ALU_ITER_IN_BITWIDTH = 14
VTA_ALU_DST_FACTOR_OUT_BITWIDTH = 11
VTA_ALU_DST_FACTOR_IN_BITWIDTH = 11
VTA_ALU_SRC_FACTOR_OUT_BITWIDTH = 11
VTA_ALU_SRC_FACTOR_IN_BITWIDTH = 11
VTA_ALU_OPCODE_BITWIDTH = 3
VTA_ALU_USE_IMM_FLAG_BITWIDTH = 1
VTA_ALU_IMM_BITWIDTH = 16

# opcode type
VTA_OPCODE_LOAD = 0
VTA_OPCODE_STORE = 1
VTA_OPCODE_GEMM = 2
VTA_OPCODE_ALU = 4