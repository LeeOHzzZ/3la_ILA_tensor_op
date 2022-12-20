""" Big Conv2D driver for HLSCNN
"""

import errno, os, sys, subprocess
import json
from math import ceil, floor
import numpy as np
from conv_layer_driver import conv_layer_driver

import tvm
from tvm import relay, runtime

SERVER_FIFO = "/home/yl29/tmp/hlscnn_ilator_server"
CLIENT_FIFO = "/home/yl29/tmp/hlscnn_ilator_client"

class Conv2DDriver(conv_layer_driver):

    def __init__(self, layer_info, addr_info, io_info):
        """Initiate the driver
        - layer_info: Conv2D layer size information (e.g., input size, kernel_size)
        - addr_info: base address of different data
        - io_info: info on writing and reading data
        """
        super().__init__(**layer_info)
        (
            self.spad_base_addr_wgt,
            self.spad_base_addr_inp,
            self.spad_base_addr_out,
        ) = addr_info
        self.wgt_wr, self.inp_wr, self.out_rd = io_info

    def produce_asm_all(self):
        if self.wgt_wr:
            self.produce_vir_mem_wr_asm_wgt(self.conv_wgt_soc_offset)
            self.produce_spad_wr_asm_wgt(
                len(self.wgt_mem), self.conv_wgt_soc_offset, self.spad_base_addr_wgt
            )
        if self.inp_wr:
            self.produce_vir_mem_wr_asm_act(self.conv_act_soc_offset)
            self.produce_spad_wr_asm_act(
                len(self.act_mem), self.conv_act_soc_offset, self.spad_base_addr_inp
            )
        self.produce_conv_layer_asm(
            self.spad_base_addr_wgt, self.spad_base_addr_inp, self.spad_base_addr_out
        )
        if self.out_rd:
            self.produce_read_asm(base_addr_out=self.spad_base_addr_out)

        self.ila_asm = {"asm": self.ila_asm}
        with open("./test/conv_ila_asm.json", "w") as fout:
            json.dump(self.ila_asm, fout, indent=2)
        with open("./test/conv_ila_data_lib.json", "w") as fout:
            json.dump(self.data_lib, fout, indent=None)

    def invoke_ila_simulator(self):
        """Invoke the hlscnn-ila simulator"""
        try:
            os.mkfifo(CLIENT_FIFO)
        except OSError as oe:
            if oe.errno != errno.EEXIST:
                raise 
        print("opening hlscnn-ilator server fifo...")
        server_fifo = open(SERVER_FIFO, "w")
        print("hlscnn-ilator server fifo opened!")
        print("starting simulation of ", self.op_name)
        server_fifo.write("start!")
        server_fifo.close()

        print("waiting hlscnn-ila client fifo response...")
        client_fifo = open(CLIENT_FIFO, "r")
        data = client_fifo.read()
        assert "finished" in data, data
        print("Received hlscnn-ila finish signal!")

    def run(self):
        subprocess.run(["mkdir", "-p", "test", "data"])

        self.collect_data_in()
        self.produce_asm_all()
        self.produce_prog_frag()
        self.invoke_ila_simulator()

        if self.out_rd:
            return self.collect_ila_result()
        else:
            return None


class Conv2DScheduler:
    """This class takes in a Conv2D layer information and a schedule that containing the
    tile sizes and loops orders of its dimensions to map to HLSCNN, and generate the
    corresponding code for each HLSCNN ilator invocations and collect the results
    """

    def __init__(self, layer_info, schedule):
        (
            self.in_chans,
            self.out_chans,
            self.in_h,
            self.in_w,
            self.k_h,
            self.k_w,
            self.stride,
            self.padding,
        ) = layer_info
        self.loop_order, self.tile_c, self.tile_k, self.tile_oh, self.tile_ow = schedule

        assert self.padding == 0, "padding size > 1 is not supported!"
        self.out_h = floor((self.in_h - self.k_h) / self.stride + 1)
        self.out_w = floor((self.in_w - self.k_w) / self.stride + 1)

        self.num_tk = ceil(self.out_chans / self.tile_k)
        self.num_tc = ceil(self.in_chans / self.tile_c)
        self.num_th = ceil(self.out_h / self.tile_oh)
        self.num_tw = ceil(self.out_w / self.tile_ow)

        # do not support spatial tiling for now
        assert self.out_h == self.tile_oh
        assert self.out_w == self.tile_ow

    def run(self, weight, input):
        # step 1: create place holder for the output
        output = np.zeros((self.out_chans, self.out_h, self.out_w))

        # assume a simple case where inputs are sliced into two tiles on input chan dimension

        # start the hlscnn-ilator simulation
        cmd = [
            "hlscnn_asm_sim_driver_server",
            "./test/conv_ila_prog_frag.json",
            "./test/conv_ila_out.json",
        ]
        subprocess.Popen(cmd) # non-blocking subprocess

        # 1st run
        tiled_wgt = weight[:, 0:8, :, :]
        tiled_inp = input[:, 0:8, :, :]
        tiled_wgt.tofile("./data/wgt.txt", sep="\n")
        tiled_inp.tofile("./data/inp.txt", sep="\n")

        layer_info = {
            "inp_size": (8, self.in_h, self.in_w),
            "out_size": (self.out_chans, self.out_h, self.out_w),
            "kernel_size": (self.out_chans, 8, self.k_h, self.k_w), 
            "stride": (self.stride, self.stride), 
            "is_bias": False,
            "bias": 0,
            "is_relu": False, 
            "is_accum": False,  
            "op_name": "hlscnn-conv2d",
        }

        addr_info = (0x04000, 0x24000, 0x34000)
        io_info = (True, True, False)
        driver = Conv2DDriver(layer_info, addr_info, io_info)
        driver.run()

        # 2nd run
        tiled_wgt = weight[:, 8:16, :, :]
        tiled_inp = input[:, 8:16, :, :]
        tiled_wgt.tofile("./data/wgt.txt", sep="\n")
        tiled_inp.tofile("./data/inp.txt", sep="\n")

        layer_info = {
            "inp_size": (8, self.in_h, self.in_w),
            "out_size": (self.out_chans, self.out_h, self.out_w),
            "kernel_size": (self.out_chans, 8, self.k_h, self.k_w), 
            "stride": (self.stride, self.stride), 
            "is_bias": False,
            "bias": 0,
            "is_relu": False, 
            "is_accum": True,  
            "op_name": "hlscnn-conv2d",
        }
        addr_info = (0x04000, 0x24000, 0x34000)
        io_info = (True, True, True)
        driver = Conv2DDriver(layer_info, addr_info, io_info)
        result = driver.run()
    
        # send stop signal to the simulator
        print("shutting down hlscnn-ilator simulation")
        server_fifo = open(SERVER_FIFO, "w")
        server_fifo.write("stop!")
        server_fifo.close()
        
        return result

def cal_single_tensor_error(result, ref):
    """
    compute mismatch within the elements of the single tensors, returns the average
    mismatches and standard deviation compared with the reference result

    Use Frobenian Norm or 2-Norm to calculate the relative mismatch
    """
    diff = result - ref
    # relative mis-match
    rmm = np.linalg.norm(diff)/np.linalg.norm(ref)
    return rmm


in_chans = 16
out_chans = 16
in_h = 32
in_w = 32
k_h = 3
k_w = 3
stride = 1
padding = 0

wgt_shape = (out_chans, in_chans, k_h, k_w)
inp_shape = (1, in_chans, in_h, in_w)
test_wgt = 0.5 * np.random.uniform(-1, 1, wgt_shape).astype("float32")
test_inp = 0.5 * np.random.uniform(-1, 1, inp_shape).astype("float32")

# test_wgt.tofile("./test/wgt.txt", sep="\n")
# test_inp.tofile("./test/inp.txt", sep="\n")

layer_info = (in_chans, out_chans, in_h, in_w, k_h, k_w, stride, padding)
schedule = ("nah", 8, out_chans, 30, 30)

test_driver = Conv2DScheduler(layer_info, schedule)
res = test_driver.run(test_wgt, test_inp)

x = relay.Var("x", relay.TensorType(inp_shape))
y = relay.Var("y", relay.TensorType(wgt_shape))
conv_func = relay.Function(
    [x, y], relay.nn.conv2d(x, y, strides=(stride, stride))
)
mod = tvm.IRModule()
mod["main"] = conv_func
with tvm.transform.PassContext():
    exe = relay.vm.compile(mod, "llvm")
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu())
    args = [test_inp, test_wgt]
    ret = vm.invoke("main", *args).asnumpy().flatten()

print(cal_single_tensor_error(res, ret))
