"""
Relay layer functions
Based on the tvm_pytorch_comparison.py provided by Steven
"""
import tvm 
import numpy as np
from tvm import relay
from tvm import IRModule
from tvm import runtime

def run_tvm_vm(expr):
  mod = tvm.IRModule.from_expr(expr)
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
  return run_tvm_vm(ln).asnumpy()
