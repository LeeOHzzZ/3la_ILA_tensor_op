"""
Relay layer functions
Based on the tvm_pytorch_comparison.py provided by Steven
"""
import tvm 
import numpy as np
from tvm import relay
from tvm import IRModule
from tvm import runtime

from .standalone_attn import luong_general_attention


def run_tvm_vm(mod):
  # mod = tvm.IRModule.from_expr(expr)
  print(mod)
  target = 'llvm'
  ctx = tvm.cpu(0)
  with tvm.transform.PassContext(opt_level=3):
      exe = relay.vm.compile(mod, target)
      vm = runtime.vm.VirtualMachine(exe, ctx)
      return vm.invoke("main")
      
def relay_layernorm(num_v, input_numpy, beta, gamma):
  const = relay.const(input_numpy)
  reshape = relay.reshape(const, (-1, num_v*16))
  b = relay.const(beta)
  g = relay.const(gamma)
  ln = relay.nn.layer_norm(reshape, gamma=g, beta=b, center=True, scale=True)
  mod = tvm.IRModule.from_expr(ln)
  return run_tvm_vm(mod).asnumpy()

def relay_attention(key_seq_len, query_seq_len, vector_size, enc_data, dec_data, wgt_data):
  mod = tvm.IRModule()
  mod["luong_attn"] = luong_general_attention(
                        batch_size=1,
                        query_units=query_seq_len,
                        key_units=key_seq_len,
                        num_units=vector_size)
  attn_var = mod.get_global_var("luong_attn")
  batch_size = 1
  dec_shape = (batch_size, query_seq_len, vector_size)
  enc_shape = (batch_size, key_seq_len, vector_size)
  weight_shape = (vector_size, vector_size)
  
  assert enc_data.shape == enc_shape, "wrong enc_data shape"
  assert dec_data.shape == dec_shape, "wrong dec_data shape"
  assert wgt_data.shape == weight_shape, "wrong weight shape"

  mod["main"] = relay.Function([], attn_var(
    relay.const(dec_data),
    relay.const(enc_data), 
    relay.const(wgt_data)))
  res = run_tvm_vm(mod)
  # return the context vector only
  return res[0].asnumpy()