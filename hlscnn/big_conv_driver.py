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
    WEIGHT_DATA_SIZE = 2
    ACTIVATION_DATA_SIZE = 2

    # related dims to different value
    WGT_DIMS = ("c", "k")
    INP_DIMS = ("c", "h", "w")

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

        self.tile_ih = (self.tile_oh - 1) * self.stride + self.k_h
        self.tile_iw = (self.tile_ow - 1) * self.stride + self.k_w

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
        print(f"In total, it needs {self.total_invokes} HLSCNN invocations!")

        # # do not support spatial tiling for now
        # assert self.out_h == self.tile_oh
        # assert self.out_w == self.tile_ow

    @staticmethod
    def start_simulator():
        # start the hlscnn-ilator simulation
        print("Starting hlscnn-ilator simulator!")
        cmd = [
            "hlscnn_asm_sim_driver_server",
            "./test/conv_ila_prog_frag.json",
            "./test/conv_ila_out.json",
        ]
        subprocess.Popen(cmd)  # non-blocking subprocess

    @staticmethod
    def stop_simulator():
        # send stop signal to the simulator
        print("shutting down hlscnn-ilator simulation")
        server_fifo = open(SERVER_FIFO, "w")
        server_fifo.write("stop!")
        server_fifo.close()

    def _get_tile_encode(self, related_dims, cntr_value):
        """This function returns the encoding of the data tile based on current
        iteration cntr and its relavent dimensions"""
        re_fn = lambda x: 1 if x in related_dims else 0
        loop_enc = [re_fn(x) for x in self.loop_order]

        return tuple(map(mul, cntr_value, loop_enc))

    def get_tile_order_from_schedule(self):
        """This function iterate over the given schedule and
        get the static order of the data tiles"""
        # list for holding the tile sequences
        t_wgt_seq = []
        t_act_seq = []
        # log for recording the shown tile
        t_wgt_log = []
        t_act_log = []

        cntr = LoopCounter(len(self.loop_bounds), self.loop_bounds)
        for i in range(self.total_invokes):
            t_wgt_id = self._get_tile_encode(self.WGT_DIMS, cntr.value)
            if t_wgt_id not in t_wgt_log:
                t_wgt_log.append(t_wgt_id)
            t_wgt_seq.append(t_wgt_log.index(t_wgt_id))

            t_act_id = self._get_tile_encode(self.INP_DIMS, cntr.value)
            if t_act_id not in t_act_log:
                t_act_log.append(t_act_id)
            t_act_seq.append(t_act_log.index(t_act_id))
            cntr.increment()

        assert sum(cntr.value) == 0
        assert len(t_wgt_seq) == self.total_invokes
        assert len(t_act_seq) == self.total_invokes

        return t_wgt_log, t_wgt_seq, t_act_log, t_act_seq

    def get_dim_stride_code(self, loop_order, loop_bounds):
        """Get the stride encoding of the dimensions
        The starting stride for all dimensions are 1. When encounter a related
        loop dimension, accumulate the stride by multiplying the previous one
        by the current loop bound.
        """
        stride_dict = {"c": 1, "k": 1, "h": 1, "w": 1}
        code = []
        for i, dim in enumerate(reversed(loop_order)):
            code.append(stride_dict[dim])
            stride_dict[dim] *= loop_bounds[-(i + 1)]
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

    def get_tiled_weight_data(self, weight, tk_idx, tc_idx):
        """Get the tiled weight data with the given tile index on its dimensions
        Assume the weight is in (out_chan, in_chan, k_h, k_w) format
        """
        # indices of out_chan dimension
        idx_kl = tk_idx * self.tile_k
        idx_kh = min(idx_kl + self.tile_k, self.out_chans)
        # indices of in_chan dimensions
        idx_cl = tc_idx * self.tile_c
        idx_ch = min(idx_cl + self.tile_c, self.in_chans)

        return weight[idx_kl:idx_kh, idx_cl:idx_ch, :, :]

    def get_tiled_input_data(self, act, tc_idx, th_idx, tw_idx):
        """Get the tiled input activation data from the given tile indices along
        different dimensions.
        Assume the input data is in (num_batch, in_chan, in_h, in_w) format

        Be aware the difference between output tensor indices and input tensor indices.
        The input arguments of tile indices are from the output tiles, and we need to use
        them to infer the input tensor boundaries.

        The higher index is exclusive, following the indexing convension of numpy/python.
        Since HLSCNN doesn't support padding, the calculating is relatively easy.
        Calculating the lower index:
            lower_idx = output_idx * stride
        Calculating the upper index:
            upper_idx = kernel_size + (output_idx - 1) * stride
        """
        # indices of in_chan dimension
        idx_cl = tc_idx * self.tile_c
        idx_ch = min(idx_cl + self.tile_c, self.in_chans)

        # indices of activation height dimension of OUTPUT tensor
        idx_ohl = th_idx * self.tile_oh
        idx_ohh = min(idx_ohl + self.tile_oh, self.out_h)
        # infer the indices of the input tensor
        idx_ihl = idx_ohl * self.stride
        idx_ihh = (idx_ohh - 1) * self.stride + self.k_h
        assert (
            idx_ihh <= self.in_h
        ), f"inferred in_h index {idx_ihh} > input dim {self.in_h}"

        # indices of activation width dimension of OUTPUT tensor
        idx_owl = tw_idx * self.tile_ow
        idx_owh = min(idx_owl + self.tile_ow, self.out_w)
        # infer the indices of the input tensor
        idx_iwl = idx_owl * self.stride
        idx_iwh = (idx_owh - 1) * self.stride + self.k_w
        assert (
            idx_iwh <= self.in_w
        ), f"inferred in_w index {idx_iwh} > input dim {self.in_w}"

        return act[:, idx_cl:idx_ch, idx_ihl:idx_ihh, idx_iwl:idx_iwh]

    def merge_tiled_result(self, out, tile_out, tk_idx, th_idx, tw_idx):
        """Merge the tiled result back to the large result if needed
        Assume the output format is (N, out_chan, out_h, out_w)
        """
        # indices of out_chan dimension
        idx_kl = tk_idx * self.tile_k
        idx_kh = min(idx_kl + self.tile_k, self.out_chans)
        # indices of activation height dimension of OUTPUT tensor
        idx_ohl = th_idx * self.tile_oh
        idx_ohh = min(idx_ohl + self.tile_oh, self.out_h)
        # indices of activation width dimension of OUTPUT tensor
        idx_owl = tw_idx * self.tile_ow
        idx_owh = min(idx_owl + self.tile_ow, self.out_w)

        out[:, idx_kl:idx_kh, idx_ohl:idx_ohh, idx_owl:idx_owh] = tile_out

        return out

    def _allocate_tensor(self, book, record, t_idx):
        """Allocate the tensor into the scrachpad memory
        book: book keeping of the current scratchpad status
        record: static information of the future access pattern
        t_idx: tensor index to be allocated

        return: the index of the allocated slot in the scratchpad
        """
        if None in book:
            idx = book.index(None)
            book[idx] = t_idx
            return idx
        else:
            next_uses = tuple(
                map(lambda x: record.index(x) if x in record else len(record), book)
            )
            idx = next_uses.index(max(next_uses))
            book[idx] = t_idx
            return idx

    def run(self, weight, input):
        """Generate the code and run the simulation with the weight and input activation
        data.
        Assume weight is in (out_chan, in_chan, kernel_h, kernel_w) format
        Assume activation is in (N_batch, in_chan, activation_h, activation_w) format
        """
        ## calculate tile sizes and related info ##
        wgt_tile_size = reduce(mul, (self.tile_k, self.tile_c, self.k_h, self.k_w, 2))
        inp_tile_size = reduce(mul, (self.tile_c, self.tile_ih, self.tile_iw, 2))
        out_tile_size = reduce(mul, (self.tile_k, self.tile_oh, self.tile_ow, 2))
        assert wgt_tile_size % 16 == 0
        assert inp_tile_size % 16 == 0
        assert out_tile_size % 16 == 0

        num_wgt_tile_avail = floor(self.SPAD_SIZE / wgt_tile_size)
        num_inp_tile_avail = floor((self.SPAD_SIZE - out_tile_size) / inp_tile_size)

        print("tile sizes: ", wgt_tile_size, inp_tile_size, out_tile_size)
        print("# of tile avail in spad: ", num_wgt_tile_avail, num_inp_tile_avail)

        # get the address for the tiles
        wgt_tile_addrs = tuple(
            i * wgt_tile_size + self.SPAD0_BASE_ADDR for i in range(num_wgt_tile_avail)
        )
        inp_tile_addrs = tuple(
            i * inp_tile_size + self.SPAD1_BASE_ADDR for i in range(num_inp_tile_avail)
        )
        out_tile_addr = inp_tile_addrs[-1] + inp_tile_size

        ## Preparation for for the loop body of hardware invocations ##
        # bookkeeping of the scratchpads
        wgt_book = [None] * num_wgt_tile_avail
        act_book = [None] * num_inp_tile_avail
        # get the static access sequence for Belady algorithm on spad replacement policy
        t_wgt_log, t_wgt_seq, t_act_log, t_act_seq = self.get_tile_order_from_schedule()

        # create a counter for loop iteration. The counter value can be used to infer
        # the tile index for each invocation
        cntr = LoopCounter(len(self.loop_bounds), self.loop_bounds)
        # create a placeholder for the output tensor
        res = np.zeros((1, self.out_chans, self.out_h, self.out_w))

        # create recorder for data movement amount
        wgt_data_mov = 0
        inp_data_mov = 0
        out_data_mov = 0
        # get the stride encoding of the loops to be used in the loops
        stride_code = self.get_dim_stride_code(self.loop_order, self.loop_bounds)

        ## start the hlscnn-ilator systemc simulation
        self.start_simulator()

        for _ in range(self.total_invokes):
            th_idx = self.get_dim_tile_idx("h", cntr.value, stride_code)
            tw_idx = self.get_dim_tile_idx("w", cntr.value, stride_code)
            tc_idx = self.get_dim_tile_idx("c", cntr.value, stride_code)
            tk_idx = self.get_dim_tile_idx("k", cntr.value, stride_code)

            print("current cntr:", cntr.cntr)

            # get the data tile indices
            t_wgt_id = t_wgt_log.index(self._get_tile_encode(self.WGT_DIMS, cntr.value))
            t_act_id = t_act_log.index(self._get_tile_encode(self.INP_DIMS, cntr.value))
            print("current tile idx: ", t_wgt_id, t_act_id)

            # determine IO requirements
            is_new_wgt = not (t_wgt_id in wgt_book)
            is_new_act = not (t_act_id in act_book)
            is_read_out = (self.loop_order[-1] is not "c") or (
                cntr.value[-1] == (self.loop_bounds[-1] - 1)
            )
            io_info = (is_new_wgt, is_new_act, is_read_out)
            # determine the data address
            if t_wgt_id in wgt_book:
                wgt_addr = wgt_tile_addrs[wgt_book.index(t_wgt_id)]
            else:
                wgt_addr = wgt_tile_addrs[
                    self._allocate_tensor(wgt_book, t_wgt_seq[cntr.cntr :], t_wgt_id)
                ]
            if t_act_id in act_book:
                inp_addr = inp_tile_addrs[act_book.index(t_act_id)]
            else:
                inp_addr = inp_tile_addrs[
                    self._allocate_tensor(act_book, t_act_seq[cntr.cntr :], t_act_id)
                ]
            addr_info = (wgt_addr, inp_addr, out_tile_addr)

            # slice out the data tiles
            # TODO: cannot put the slicing into the next conditional block (is_new_wgt/act) below
            # because the layer_info needs the updated conv2d tensor shape information
            t_wgt = self.get_tiled_weight_data(weight, tk_idx, tc_idx)
            t_inp = self.get_tiled_input_data(input, tc_idx, th_idx, tw_idx)

            # dump the tiled data to files
            if is_new_wgt:
                t_wgt.tofile("./data/wgt.txt", sep="\n")
                wgt_data_mov += t_wgt.size * self.WEIGHT_DATA_SIZE
            if is_new_act:
                t_inp.tofile("./data/inp.txt", sep="\n")
                inp_data_mov += t_inp.size * self.ACTIVATION_DATA_SIZE

            t_out_h = floor((t_inp.shape[2] - self.k_h) / self.stride + 1)
            t_out_w = floor((t_inp.shape[3] - self.k_w) / self.stride + 1)
            is_accum = (self.loop_order[-1] is "c") and (cntr.value[-1] > 0)
            layer_info = {
                "inp_size": (t_inp.shape[1], t_inp.shape[2], t_inp.shape[3]),
                "out_size": (t_wgt.shape[0], t_out_h, t_out_w),
                "kernel_size": (t_wgt.shape[0], t_wgt.shape[1], self.k_h, self.k_w),
                "stride": (self.stride, self.stride),
                "is_bias": False,
                "bias": 0,
                "is_relu": False,
                "is_accum": is_accum,
                "op_name": f"hlscnn-tiled-conv2d-{cntr.cntr}",
            }
            print("io_info:", io_info)
            print("addr_info:", addr_info)
            print("layer_info:", layer_info)

            # call the code generation for the tile conv2d operator
            driver = Conv2DDriver(layer_info, addr_info, io_info)
            tile_result = driver.run()

            # merge the results if needed
            if is_read_out:
                assert tile_result is not None
                tile_result = np.array(tile_result).reshape(
                    (1, t_wgt.shape[0], t_out_h, t_out_w)
                )
                res = self.merge_tiled_result(res, tile_result, tk_idx, th_idx, tw_idx)
                out_data_mov += tile_result.size * self.ACTIVATION_DATA_SIZE

            cntr.increment()

        self.stop_simulator()

        print(
            f"""
        Data Movement Summary:
        Total weight data write: {wgt_data_mov} bytes,
        Total input data write: {inp_data_mov} bytes,
        Total output read: {out_data_mov} bytes,
        Total Data Movement: {wgt_data_mov + inp_data_mov + out_data_mov} bytes.
        """
        )

        return res


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


def test():
    in_chans = 8
    out_chans = 64
    in_h = 226
    in_w = 226
    k_h = 3
    k_w = 3
    stride = 1
    padding = 0

    wgt_shape = (out_chans, in_chans, k_h, k_w)
    inp_shape = (1, in_chans, in_h, in_w)
    test_wgt = 0.5 * np.random.uniform(-1, 1, wgt_shape).astype("float32")
    test_inp = 0.5 * np.random.uniform(-1, 1, inp_shape).astype("float32")

    layer_info = (in_chans, out_chans, in_h, in_w, k_h, k_w, stride, padding)
    schedule = ("chwk", 8, 8, 56, 56)
    test_driver = Conv2DScheduler(layer_info, schedule)
    res = test_driver.run(test_wgt, test_inp)

    x = relay.Var("x", relay.TensorType(inp_shape))
    y = relay.Var("y", relay.TensorType(wgt_shape))
    conv_func = relay.Function([x, y], relay.nn.conv2d(x, y, strides=(stride, stride)))
    mod = tvm.IRModule()
    mod["main"] = conv_func
    with tvm.transform.PassContext():
        exe = relay.vm.compile(mod, "llvm")
        vm = runtime.vm.VirtualMachine(exe, tvm.cpu())
        args = [test_inp, test_wgt]
        ret = vm.invoke("main", *args).asnumpy()

    print(f"{cal_single_tensor_error(res, ret):.5%}")


if __name__ == "__main__":
    test()
