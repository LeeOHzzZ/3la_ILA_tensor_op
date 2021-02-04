"""
Relay LSTM reference
Based on the flexnlp_layers.py provided by Steven
"""
import tvm
import numpy as np
from tvm import relay
from tvm import IRModule
from tvm.contrib import graph_runtime
from tvm.relay.testing.lstm import lstm_cell

def time_distribute(inp, produce_inner_expr, time_dim_width, time_axis=1):
    """
    Given an input tensor `inp`, splits it along the specified time axis
    into `time_dim_width` slices (`time_dim_width` should match the length of the time axis)
    and calls `produce_inner_expr` on the time slices.

    `produce_inner_expr` takes an array of slices from `inp`
    where the time dimension has been eliminated
    and returns a tuple of `time_dim_width` outputs,
    which are then combined back together,
    stacked on the time axis.

    For example, given a tensor of shape (1, 10, 3, 224, 224)
    where 1 is the time axis, this function would produce 10
    slices of shape (1, 3, 224, 224). `produce_inner_expr`
    would be called on them and return a tuple of 10 tensors
    (let's suppose of the same shape), which are then stacked
    back together into a tensor of shape (1, 10, 3, 224, 224)
    """

    split_var = relay.Var('splits')
    # split the input on the time axis
    # and squeeze away the remaining time dimension (which will have a shape of 1)
    split_values = [
        relay.squeeze(
            relay.TupleGetItem(split_var, i),
            axis=[time_axis]
        )
        for i in range(time_dim_width)
    ]
    return relay.Let(
        split_var,
        relay.split(inp, time_dim_width, time_axis).astuple(),
        relay.stack(
            produce_inner_expr(split_values),
            axis=time_axis
        )
    )

def lstm_layer(inp, num_hidden,
               i2h_weight, i2h_bias, h2h_weight, h2h_bias,
               init_states,
               time_dim_width, time_axis=1):
    lstm_var = relay.Var('lstm')
    lstm_func = lstm_cell(num_hidden, batch_size=1, name='lstm_func')

    # LSTM needs the value of the previous split
    def lstm_body(time_splits):
        builder = relay.ScopeBuilder()
        prev_states = init_states
        cell_outputs = []

        for i, split in enumerate(time_splits):
            cell = builder.let(f'cell_out_{i}',
                               lstm_var(split, prev_states,
                                        i2h_weight, i2h_bias,
                                        h2h_weight, h2h_bias))
            cell_ret = builder.let('cell_ret_{i}', relay.TupleGetItem(cell, 0))
            cell_outputs.append(cell_ret)
            prev_states = builder.let(f'cell_states_{i}', relay.TupleGetItem(cell, 1))

        builder.ret(relay.Tuple(cell_outputs))
        return builder.get()

    return relay.Let(
        lstm_var, lstm_func,
        time_distribute(inp, lstm_body,
                        time_dim_width, time_axis)
    )

def relay_lstm_ref(num_v_in, num_v_out, num_ts, 
                   input, wgt_i, wgt_h, bias_i, bias_h):
    # need to instantiate weights and bias to run
    assert num_v_in == num_v_out

    dtype = 'float32'
    num_hidden = num_v_out * 16
    batch_size = 1
    time_dim_width = num_ts
    # now set up all the input values
    builder = relay.ScopeBuilder()
    input_shape = (batch_size, time_dim_width, num_hidden)
    inner_input_shape = (batch_size, num_hidden)
    weight_shape = (4 * num_hidden, num_hidden)
    bias_shape = (4 * num_hidden, )

    # inp = builder.let('inp', relay.const(np.random.rand(*input_shape)))
    # i2h_weight = builder.let('i2h_weight', relay.const(np.random.rand(*weight_shape)))
    # h2h_weight = builder.let('h2h_weight', relay.const(np.random.rand(*weight_shape)))
    # i2h_bias = builder.let('i2h_bias', relay.const(np.random.rand(*bias_shape)))
    # h2h_bias = builder.let('h2h_bias', relay.const(np.random.rand(*bias_shape)))
    init_states = builder.let('init_states', relay.Tuple([
        relay.zeros(inner_input_shape, dtype),
        relay.zeros(inner_input_shape, dtype)
    ]))

    # convert np arrays to tvm.nd.array
    inp = input.reshape((num_ts, 16*num_v_in))
    inp = np.expand_dims(inp, axis=0)
    inp = builder.let('inp', relay.const(inp))
    i2h_weight = builder.let('i2h_weight', relay.const(wgt_i))
    h2h_weight = builder.let('h2h_weight', relay.const(wgt_h))
    i2h_bias = builder.let('i2h_bias', relay.const(bias_i))
    h2h_bias = builder.let('h2h_bias', relay.const(bias_h))


    builder.ret(
      lstm_layer(
          inp, num_hidden,
          i2h_weight, i2h_bias, h2h_weight, h2h_bias,
          init_states, time_dim_width, time_axis=1)
    )
    mod = IRModule.from_expr(builder.get())
#    print(mod)

    target = 'llvm'
    args = ()
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        exe = relay.vm.compile(mod, target)
        vm = tvm.runtime.vm.VirtualMachine(exe, ctx)
        out = vm.invoke("main", *args)
    
    out_np = out.asnumpy()
    # print(out_np.shape)
    # print(out_np)
    return out_np



# if __name__ == '__main__':
#     relay_lstm_ref()
