from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import numpy as np
from tvm import rpc
# from tvm.contrib import util
from tvm.relay.op.contrib import ilaflex

from utils import tool

# define the graph
dtype="float32"
m = 64
n = 64

shape1 = tvm.relay.TensorType((n, m), dtype=dtype)
shape2 = tvm.relay.TensorType((n, m), dtype=dtype)
shape3 = tvm.relay.TensorType((n, ), dtype=dtype)
x = tvm.relay.var("x", shape1)
w = tvm.relay.var("w", shape2)
b = tvm.relay.var("b", shape3)
matmul = tvm.relay.nn.dense(x, w)
final = tvm.relay.nn.bias_add(matmul, b)

mod = tvm.ir.IRModule.from_expr(final)
print(mod)


# pattern matching
pattern_table = ilaflex.pattern_table()
mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
mod = tvm.relay.transform.AnnotateTarget(["ilaflex"])(mod) 
mod = tvm.relay.transform.PartitionGraph()(mod) 

print("[Python] Compile with the 3LA extension")
target = tvm.target.create('llvm')
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = tvm.relay.build(mod, target)

##
## execute
##
from tvm.contrib import graph_runtime
ctx = tvm.cpu()
runtime_exec = graph_runtime.create(graph, lib, ctx)

coef = 0.2
x_np = coef * np.random.uniform(0, 1, size=(n, m)).astype(np.float32)
y_np = coef * np.random.uniform(0, 1, size=(n, m)).astype(np.float32)
z_np = coef * np.random.uniform(0, 1, size=(n,)).astype(np.float32)

ref = np.add(np.matmul(x_np, np.transpose(y_np)), z_np)
# x_np = coef * np.random.random_sample((n, m), dtype = np.float32)
# x_np = np.array([[1, 2], [3, 4]], dtype = np.float32)
# y_np = coef * np.random.random_sample((n, m), dtype = np.float32)
# z_np = coef * np.random.random_sample((n,), dtype = np.float32)

x_tvm = tvm.nd.array(x_np, ctx=ctx)
y_tvm = tvm.nd.array(y_np, ctx=ctx)
z_tvm = tvm.nd.array(z_np, ctx=ctx)

print("[Python] Execute the compiled model")
runtime_exec.set_input(0, x_tvm)
runtime_exec.set_input(1, y_tvm)
runtime_exec.set_input(2, z_tvm)
runtime_exec.set_input(**params)
runtime_exec.run()

output = runtime_exec.get_output(0).asnumpy()
output = output.astype(np.float32)
print("[Python] Done")

tl = tool()
err_out, err_ref = tl.cal_error(output, ref)
# print(err_out, err_ref)
print("relative error: {:5.5%} vs. output, {:5.5%} vs. ref".format(err_out, err_ref))
print('output: {}'.format(output.shape))
print(output)
print('===============')
print('ref: {}'.format(ref.shape))
print(ref)
