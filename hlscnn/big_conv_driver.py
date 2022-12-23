""" Big Conv2D driver for HLSCNN
"""

import errno, os, sys, subprocess
import json
from math import ceil, floor
from operator import mul
from functools import reduce
import numpy as np
from conv_layer_driver import conv_layer_driver
from utils import LoopCounter

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

    SPAD_SIZE = 0x20000
    SPAD0_BASE_ADDR = 0x04000
    SPAD1_BASE_ADDR = SPAD0_BASE_ADDR + SPAD_SIZE

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
        assert self.tile_c % 8 == 0
        assert self.tile_k % 8 == 0

        self.out_h = floor((self.in_h - self.k_h) / self.stride + 1)
        self.out_w = floor((self.in_w - self.k_w) / self.stride + 1)

        self.tile_ih = (self.out_h - 1) * self.stride + self.k_h
        self.tile_iw = (self.out_w - 1) * self.stride + self.k_w

        self.num_tc = ceil(self.in_chans / self.tile_c)
        self.num_tk = ceil(self.out_chans / self.tile_k)
        self.num_th = ceil(self.out_h / self.tile_oh)
        self.num_tw = ceil(self.out_w / self.tile_ow)
        self.tnum_dict = {
            "c": self.num_tc,
            "k": self.num_tk,
            "h": self.num_th,
            "w": self.num_tw,
        }
        self.total_invokes = reduce(
            mul, (self.num_tc, self.num_tk, self.num_th, self.num_tw)
        )
        self.loop_bounds = tuple(self.tnum_dict[i] for i in self.loop_order)

        # # do not support spatial tiling for now
        # assert self.out_h == self.tile_oh
        # assert self.out_w == self.tile_ow

    def get_tile_order_from_schedule(self):
        """This function iterate over the given schedule and get the static order of the data tiles"""
        re_fn = lambda x, dims: 1 if x in dims else 0
        loop_enc_wgt = [re_fn(x, ("c", "k")) for x in self.loop_order]
        loop_enc_act = [re_fn(x, ("c", "h", "w")) for x in self.loop_order]
        # list for holding the tile sequences
        t_wgt_seq = []
        t_act_seq = []
        # log for recording the shown tile
        t_wgt_log = []
        t_act_log = []

        cntr = LoopCounter(len(self.loop_bounds), self.loop_bounds)
        for i in range(self.total_invokes):
            it_list = cntr.value
            t_wgt_id = tuple(map(mul, it_list, loop_enc_wgt))
            if t_wgt_id not in t_wgt_log:
                t_wgt_log.append(t_wgt_id)
            t_wgt_seq.append(t_wgt_log.index(t_wgt_id))
            t_act_id = tuple(map(mul, it_list, loop_enc_act))
            if t_act_id not in t_act_log:
                t_act_log.append(t_act_id)
            t_act_seq.append(t_act_log.index(t_act_id))
            cntr.increment()
        assert sum(cntr.value) == 0
        assert len(t_wgt_seq) == self.total_invokes
        assert len(t_act_seq) == self.total_invokes

        return t_wgt_log, t_wgt_seq, t_act_log, t_act_seq

    def get_dim_stride_code(self, loop_order, loop_bounds):
        """Get the stride encoding of the dimensions"""
        stride_dict = {"c": 1, "k": 1, "h": 1, "w": 1}
        code = []
        for i, dim in enumerate(reversed(loop_order)):
            code.append(stride_dict[dim])
            stride_dict[dim] *= loop_bounds[-(i+1)]
        return tuple(code[::-1])

    def get_dim_tile_idx(self, dim, iter_code, loop_stride_code):
        """Return the tile idx of the given dimension based on the loop information
        The tile index of can be acqure by the sum of the produce of three encodeing
        - relevance code: code that indicate which loops are related to this dimension
        - iteration code: code that contains the current loop information
        - loop stride code: code that contains the stride size of each loop on their
            related dimensions
        e.g., for a given dimension:
              - rev code:           (0, 1, 1, 0)
              - iteration code:     (2, 3, 1, 2)
              - stride code:        (1, 4, 1, 1)
              the index the current tile on this dimension is 1*3*4 + 1*1*1 = 13
        """
        rev_code = tuple(1 if i == dim else 0 for i in self.loop_order)
        fn = lambda x, y, z: x * y * z
        idx = sum((map(fn, rev_code, iter_code, loop_stride_code)))
        return idx

    def run(self, weight, input):
        wgt_tile_size = reduce(mul, (self.tile_k, self.tile_c, self.k_h, self.k_w, 2))
        inp_tile_size = reduce(mul, (self.tile_c, self.tile_ih, self.tile_iw, 2))
        out_tile_size = reduce(mul, (self.tile_k, self.tile_oh, self.tile_ow, 2))
        num_wgt_tile_avail = floor(self.SPAD_SIZE / wgt_tile_size)
        num_inp_tile_avail = floor((self.SPAD_SIZE - out_tile_size) / inp_tile_size)

        assert wgt_tile_size % 16 == 0
        assert inp_tile_size % 16 == 0
        assert out_tile_size % 16 == 0

        # get the address for the tiles
        wgt_tile_addrs = tuple(
            i * wgt_tile_size + self.SPAD0_BASE_ADDR for i in range(num_wgt_tile_avail)
        )
        inp_tile_addrs = tuple(
            i * inp_tile_size + self.SPAD1_BASE_ADDR for i in range(num_inp_tile_avail)
        )
        out_tile_addr = inp_tile_addrs[-1] + inp_tile_size
        # bookkeeping of the scratchpads
        wgt_book = tuple(None for i in range(num_wgt_tile_avail))
        act_book = tuple(None for i in range(num_inp_tile_avail))
        # get the static access sequence of the tiles
        t_wgt_log, t_wgt_seq, t_act_log, t_act_seq = self.get_tile_order_from_schedule()
        # get the stride encoding of the loops
        stride_code = self.get_dim_stride_code(self.loop_order, self.loop_bounds)

        cntr = LoopCounter(len(self.loop_bounds), self.loop_bounds)

        for i in range(self.total_invokes):
            th_idx = self.get_dim_tile_idx("h", cntr.value, stride_code)
            tw_idx = self.get_dim_tile_idx("w", cntr.value, stride_code)
            tc_idx = self.get_dim_tile_idx("c", cntr.value, stride_code)
            tk_dix = self.get_dim_tile_idx("k", cntr.value, stride_code)
            cntr.increment()
            print(th_idx, tw_idx, tc_idx, tk_dix)

        # for l0 in range(self.tnum_dict[self.loop_order[0]]):
        #     for l1 in range(self.tnum_dict[self.loop_order[1]]):
        #         for l2 in range(self.tnum_dict[self.loop_order[2]]):
        #             for l3 in range(self.tnum_dict[self.loop_order[3]]):
        #                 it_list = (l0, l1, l2, l3)

        pass

    def run_test(self, weight, input):
        # step 1: create place holder for the output
        output = np.zeros((self.out_chans, self.out_h, self.out_w))

        # assume a simple case where inputs are sliced into two tiles on input chan dimension

        # start the hlscnn-ilator simulation
        cmd = [
            "hlscnn_asm_sim_driver_server",
            "./test/conv_ila_prog_frag.json",
            "./test/conv_ila_out.json",
        ]
        subprocess.Popen(cmd)  # non-blocking subprocess

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
    rmm = np.linalg.norm(diff) / np.linalg.norm(ref)
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
schedule = ("hwck", 8, 8, 15, 15)
test_driver = Conv2DScheduler(layer_info, schedule)
test_driver.run(test_wgt, test_inp)

# schedule = ("nah", 8, out_chans, 30, 30)
# test_driver = Conv2DScheduler(layer_info, schedule)
# res = test_driver.run(test_wgt, test_inp)

# x = relay.Var("x", relay.TensorType(inp_shape))
# y = relay.Var("y", relay.TensorType(wgt_shape))
# conv_func = relay.Function([x, y], relay.nn.conv2d(x, y, strides=(stride, stride)))
# mod = tvm.IRModule()
# mod["main"] = conv_func
# with tvm.transform.PassContext():
#     exe = relay.vm.compile(mod, "llvm")
#     vm = runtime.vm.VirtualMachine(exe, tvm.cpu())
#     args = [test_inp, test_wgt]
#     ret = vm.invoke("main", *args).asnumpy().flatten()

# print(cal_single_tensor_error(res, ret))
