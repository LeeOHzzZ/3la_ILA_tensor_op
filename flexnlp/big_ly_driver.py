""" Big Conv2D driver for HLSCNN
"""

import errno, os, sys, subprocess
import json
from math import ceil, floor
from operator import mul
from functools import reduce
import numpy as np

# from src.converter import Converter as cvtr
from src._ts_asm_converter import ts_asm_converter as asm_cvtr
from src._asm_prog_frag_converter import asm_prog_frag_converter as pg_fg_cvtr
from linear_layer_driver import linear_layer_driver
from utils import LoopCounter

SERVER_FIFO = "/home/yl29/tmp/flexasr_ilator_server"
CLIENT_FIFO = "/home/yl29/tmp/flexasr_ilator_client"


class FlexASRLinearLayerDriver(linear_layer_driver):
    def __init__(self, layer_info, addr_info, io_info):
        super().__init__(**layer_info)
        self.pe_wgt_addr, self.gb_inp_addr, self.gb_out_addr = addr_info
        self.wgt_wr, self.inp_wr, self.out_rd = io_info
        # does not support biased linear layer yet
        assert self.is_bias == False

    def collect_input_data(self):
        self.collect_data_wgt()
        self.collect_data_inp()

    def produce_data_lib(self):
        assert self.dtype == "float32"
        wgt_q, bias_wgt = self.tl.get_adpfloat_bias(
            self.wgt, self.ADPTFLOAT_N_BITS, self.ADPTFLOAT_N_EXP, self.ADPTBIAS
        )
        self.wgt = wgt_q
        self.bias_wgt = int(bias_wgt + self.ADPTFLOAT_OFFSET)

        inp_q, bias_inp = self.tl.get_adpfloat_bias(
            self.inp, self.ADPTFLOAT_N_BITS, self.ADPTFLOAT_N_EXP, self.ADPTBIAS
        )
        self.inp = inp_q
        self.bias_inp = int(bias_inp + self.ADPTFLOAT_OFFSET)

        self.bias_b = 0
        self.bias_act = 3
        print(f"{self.bias_wgt}::{self.bias_b}::{self.bias_inp}::{self.bias_act}")

        # init data-lib param
        self.data_lib = {
            "gb_num_vector_in": self.num_v_in,
            "gb_num_vector_out": self.num_v_out,
            "adpbias_wgt": self.bias_wgt,
            "adpbias_inp": self.bias_inp,
            "adpbias_bias": self.bias_b,  # hardcoded it to 0
            "adpbias_pe_act": self.bias_act,
            "w0_num_tile": int(self.num_v_in * self.num_v_out),
        }

        if self.wgt_wr:
            # convert the weight tensor layout to FlexASR tiling format
            wgt_qt = self.tl.wgt_tiling(wgt_q, self.num_v_in, self.num_v_out)
            wgt_qt.tofile("./test/wgt_qt.tmp", sep="\n")
            self.tl.call_float_adpt_v_cvtr(
                "./test/wgt_qt.tmp", self.bias_wgt, "./test/wgt_qt_av.tmp"
            )
            with open("./test/wgt_qt_av.tmp", "r") as fin:
                wgt_v_list = fin.read().splitlines()
            assert len(wgt_v_list) % 16 == 0
            self.data_lib = self.tl.wgt_to_data_lib(
                wgt_v_list, "w0", self.num_v_in * self.num_v_out, self.data_lib
            )
        if self.inp_wr:
            inp_q.tofile("./test/inp_q.tmp", sep="\n")
            self.tl.call_float_adpt_v_cvtr(
                "./test/inp_q.tmp", self.bias_inp, "./test/inp_q_av.tmp"
            )
            with open("./test/inp_q_av.tmp", "r") as fin:
                inp_v_list = fin.read().splitlines()
            assert len(inp_v_list) == self.num_v_in * self.num_ts
            for t in range(self.num_ts):
                self.data_lib = self.tl.vector_to_data_lib(
                    inp_v_list[t * self.num_v_in : (t + 1) * self.num_v_in],
                    f"ts_{t}",
                    self.num_v_in,
                    self.data_lib,
                )

    def produce_prog_frag(self, write_only=False):
        ila_cvtr = asm_cvtr([], self.data_lib)
        asm = []
        # need to manually set the gb large memory base addrs
        ila_cvtr.set_gb_large_buf_mem_base(0, self.gb_inp_addr)
        ila_cvtr.set_gb_large_buf_mem_base(1, self.gb_out_addr)
        # generate memory write instructions
        if self.wgt_wr:
            unit_asm = {"wgt_idx": "w0"}
            asm += ila_cvtr.gen_store_wgt(unit_asm, addr_offset=self.pe_wgt_addr)
        if self.inp_wr:
            for i in range(self.num_ts):
                unit_asm = {"timestep_idx": f"ts_{i}", "idx": i, "mem_idx": 0}
                asm += ila_cvtr.gen_store_act(unit_asm)

        # configuring the linear layer operation
        #   configuring PEs
        pe_num_v_out = int(self.num_v_out / 4)
        for pe_idx in range(4):
            asm.append(
                {
                    "name": "pe_cfg_rnn_layer_sizing",
                    "pe_idx": pe_idx,
                    "is_zero": 0,
                    "is_cluster": 0,
                    "is_bias": self.is_bias,
                    "num_mngr": 1,
                    "num_v_out": pe_num_v_out,
                }
            )
            asm.append(
                {
                    "name": "pe_cfg_mngr",
                    "pe_idx": pe_idx,
                    "mngr_idx": 1,
                    "is_zero": 0,
                    "adpbias_wgt": self.data_lib["adpbias_wgt"],
                    "adpbias_bias": self.data_lib["adpbias_bias"],
                    "adpbias_inp": self.data_lib["adpbias_inp"],
                    "num_v_in": self.num_v_in,
                    "base_wgt": self.pe_wgt_addr >> 4,  # TODO: vector level
                    "base_bias": 0,  # should not be used
                    "base_inp": 0,  # should be only 1 input within PE each time
                }
            )
            asm.append(
                {
                    "name": "pe_cfg_act_mngr",
                    "pe_idx": pe_idx,
                    "is_zero": 0,
                    "adpfloat_bias": self.data_lib["adpbias_pe_act"],
                    "num_insn": 2,
                    "num_v_out": pe_num_v_out,
                    "buf_base": 0,
                    "out_base": pe_idx * pe_num_v_out,
                }
            )
            asm.append(
                {
                    "name": "pe_cfg_act_v",
                    "pe_idx": pe_idx,
                    "v_idx": 1,
                    "insn_0": "0x30",
                    "insn_1": "0x40",
                }
            )
        #   configure GB
        asm.append(
            {
                "name": "cfg_mmngr_gb_large",
                "base_0": hex(
                    self.gb_inp_addr >> 4
                ),  # convert it to vector level address
                "num_v_0": self.num_v_in,
                "base_1": hex(
                    self.gb_out_addr >> 4
                ),  # convert it to vector level address
                "num_v_1": self.num_v_out,
            }
        )
        asm.append(
            {
                "name": "cfg_gb_ctrl",
                "mode": 0,
                "is_rnn": 0,
                "mem_id_i": 0,
                "mem_id_o": 1,
                "num_v_i": self.num_v_in,
                "num_v_o": self.num_v_out,
                "num_ts": self.num_ts,
            }
        )
        # trigger the operation
        asm.append(
            {
                "name": "start",
                "op": 1,
            }
        )

        if self.out_rd and not write_only:
            asm.append({"name": "wait_irq"})
            for i in range(self.num_ts):
                unit_asm = {"mem_idx": 1, "ts_idx": i}
                asm += ila_cvtr.gen_load_act(unit_asm)

        prog_frag_cvtr = pg_fg_cvtr(asm, self.data_lib)
        prog_frag = prog_frag_cvtr.to_ila_prog_frag()
        with open("./test/ly_prog_frag_in.json", "w") as fout:
            json.dump({"program fragment": prog_frag}, fout, indent=4)

    def invoke_ilator_simulation(self):
        """Invoke the flexasr-ila simulator"""
        try:
            os.mkfifo(CLIENT_FIFO)
        except OSError as oe:
            if oe.errno != errno.EEXIST:
                raise
        print("opening flexasr-ilator server fifo...")
        server_fifo = open(SERVER_FIFO, "w")
        print("flexasr-ilator server fifo opened!")
        print("starting simulation of ", self.op_name)
        server_fifo.write("start!")
        server_fifo.close()

        print("waiting flexasr-ila client fifo response...")
        client_fifo = open(CLIENT_FIFO, "r")
        data = client_fifo.read()
        assert "finished" in data, data
        print("Received flexasr-ila finish signal!")

    def run(self):
        # create folders for intermediate data
        subprocess.run(["mkdir", "-p", "data", "test"])
        self.collect_input_data()
        self.produce_data_lib()
        self.produce_prog_frag()
        self.invoke_ilator_simulation()

        if self.out_rd:
            result = self.tl.collect_axi_out_new(
                in_path="./test/flexasr_adpf_result.txt",
                out_path="./test/result.tmp",
                mem_base=self.gb_out_addr,
                num_ts=self.num_ts,
                num_vo=self.num_v_out,
                bias=self.bias_act,
            )
            return result


def test_big_driver():
    layer_info = {
        "num_v_in": 4,
        "num_v_out": 4,
        "num_timestep": 5,
        "is_bias": False,
        "op_name": "flexasr_linear_layer_test",
        "dtype": "float32",
    }
    addr_info = (0, 0, 0x01000)
    io_info = (True, True, True)
    driver = FlexASRLinearLayerDriver(layer_info, addr_info, io_info)
    result = driver.run()
    print(result)


class FlexASRLinearLayerScheduler:
    """This class takes in a Linear Layer information and a schedule that containing the
    tile sizes and loop orders of its dimensions to map to FlexASR linear layer operation,
    and generate the corresponding code for each FlexASR ilator invocations and collect
    the final results
    """
    # We can actually set the value here to be smaller than the FlexASR original spec
    # for faster simulation to test the schedule
    NUM_PE = 4  # 4 PE in FlexASR
    GBCORE_LARGE_BUF_SIZE = 0x10000  # 1MB -> 64KB
    PECORE_LARGE_BUF_SIZE = 0x10000  # 1MB per PE -> 64KB
    # related dims to different value
    WGT_DIMS = ("x", "y")
    INP_DIMS = ("t", "x")

    def __init__(self, layer_info, schedule):
        (
            self.dim_batch,
            self.dim_x,
            self.dim_y,
        ) = layer_info
        self.loop_order, self.tile_batch, self.tile_x, self.tile_y = schedule

        assert self.dim_x % 16 == 0
        assert self.dim_y % 64 == 0
        assert self.tile_x % 16 == 0
        assert self.tile_y % 64 == 0

        self.num_tb = ceil(self.dim_batch / self.tile_batch)
        self.num_tx = ceil(self.dim_x / self.tile_x)
        self.num_ty = ceil(self.dim_y / self.tile_y)
        self.tnum_dict = {
            "t": self.num_tb,
            "x": self.num_tx,
            "y": self.num_ty,
        }
        self.total_invokes = reduce(mul, (self.num_tb, self.num_tx, self.num_ty))
        self.loop_bounds = tuple(self.tnum_dict[i] for i in self.loop_order)
        print(f"In total, it needs {self.total_invokes} FlexASR invocations!")

    @staticmethod
    def start_simulator():
        # start the flexasr-ilator simulation
        print("Starting flexasr-ilator simulator!")
        cmd = [
            "flex_asm_sim_driver_server",
            "./test/ly_prog_frag_in.json",
            "./test/flexasr_adpf_result.txt",
        ]
        subprocess.Popen(cmd)  # non-blocking subprocess

    @staticmethod
    def stop_simulator():
        # send stop signal to the simulator
        print("shutting down the flexasr-ilator simulation")
        server_fifo = open(SERVER_FIFO, "w")
        server_fifo.write("stop!")
        server_fifo.close()

    def pad_to(self, num, unit):
        """pad the input num up to multiple of units"""
        return ceil(num / unit) * unit

    def _get_tile_encode(self, related_dims, cntr_value):
        """This function returns the encoding of the data tile based on current
        iteration cntr and its relavent dimensions"""
        re_fn = lambda x: 1 if x in related_dims else 0
        loop_enc = [re_fn(x) for x in self.loop_order]

        return tuple(map(mul, cntr_value, loop_enc))

    def get_tile_order_from_schedule(self):
        """This function iterate over the given scehdule and get the static order
        of the data tiles
        """
        # list for the holding the tile sequences
        t_wgt_seq = []
        t_act_seq = []
        # log for recording the shown tile
        t_wgt_log = []
        t_act_log = []

        cntr = LoopCounter(len(self.loop_bounds), self.loop_bounds)
        for _ in range(self.total_invokes):
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
        stride_dict = {"t": 1, "x": 1, "y": 1}
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

    def get_tiled_weight_data(self, weight, ty_idx, tx_idx):
        """Get the tiled weight data from the given tile index on its dimensions
        Assume the weight is in (dim_y, dim_x) format"""
        # indices of the y dimension
        idx_yl = ty_idx * self.tile_y
        idx_yh = min(idx_yl + self.tile_y, self.dim_y)
        # indices of the x dimension
        idx_xl = tx_idx * self.tile_x
        idx_xh = min(idx_xl + self.tile_x, self.dim_x)

        return weight[idx_yl:idx_yh, idx_xl:idx_xh]

    def get_tiled_input_data(self, input, tb_idx, tx_idx):
        """Get the tiled input data with the given tile indices
        Assume the input is in (dim_t, dim_x) format"""
        # indices of the batch dimension
        idx_bl = tb_idx * self.tile_batch
        idx_bh = min(idx_bl + self.tile_batch, self.dim_batch)
        # indices of the x dimension
        idx_xl = tx_idx * self.tile_x
        idx_xh = min(idx_xl + self.tile_x, self.dim_x)

        return input[idx_bl:idx_bh, idx_xl:idx_xh]

    def merge_tiled_result(self, res, tile_out, tb_idx, ty_idx):
        """Merge the tiled result back to the large result
        Assume the output format is (dim_batch, dim_y)
        """
        # indices of the batch dimension
        idx_bl = tb_idx * self.tile_batch
        idx_bh = min(idx_bl + self.tile_batch, self.dim_batch)
        # indices of the y dimension
        idx_yl = ty_idx * self.tile_y
        idx_yh = min(idx_yl + self.tile_y, self.dim_y)

        res[idx_bl:idx_bh, idx_yl:idx_yh] += tile_out

        return res

    def run(self, weight, input):
        """Generate the code and run the simulation with the weight and input data"""
        # calculate tile sizes and related info
        wgt_tile_size = self.tile_x * self.tile_y
        inp_tile_size = self.tile_batch * self.tile_x
        out_tile_size = self.tile_batch * self.tile_y
        # need to pad the inp tile size to group of 16 timesteps
        padded_inp_tile_size = self.pad_to(self.tile_batch, 16) * self.tile_x
        padded_out_tile_size = self.pad_to(self.tile_batch, 16) * self.tile_y

        num_wgt_tile_avail = floor(
            self.PECORE_LARGE_BUF_SIZE * self.NUM_PE / wgt_tile_size
        )
        num_inp_tile_avail = floor(
            (self.GBCORE_LARGE_BUF_SIZE - padded_out_tile_size) / padded_inp_tile_size
        )

        print("tile sizes: ", wgt_tile_size, inp_tile_size, out_tile_size)
        print("padded inp/out tile size: ", padded_inp_tile_size, padded_out_tile_size)
        print("# of tile avail in memory: ", num_wgt_tile_avail, num_inp_tile_avail)

        # get the address for the tiles
        # just need the relative address in each memory
        wgt_tile_addrs = tuple(i * wgt_tile_size for i in range(num_wgt_tile_avail))
        inp_tile_addrs = tuple(
            i * padded_inp_tile_size for i in range(num_inp_tile_avail)
        )
        out_tile_addr = inp_tile_addrs[-1] + padded_inp_tile_size
        # print("wgt_tile_addrs:", list(map(hex, wgt_tile_addrs)))
        # print("inp_tile_addrs:", list(map(hex, inp_tile_addrs)))
        # print("out_tile_addrs:", hex(out_tile_addr))

        ## Preparation for for the loop body of hardware invocations ##
        # bookkeeping of the scratchpads
        wgt_book = [None] * num_wgt_tile_avail
        act_book = [None] * num_inp_tile_avail
        # get the static access sequence for Belady algorithm on spad replacement policy
        t_wgt_log, t_wgt_seq, t_act_log, t_act_seq = self.get_tile_order_from_schedule()

        print("t_wgt_seq:", t_wgt_seq)
        print("t_act_seq:", t_act_seq)

        # create a counter for loop iteration. The counter value can be used to infer
        # the tile index for each invocation
        cntr = LoopCounter(len(self.loop_bounds), self.loop_bounds)
        # create a placeholder for the output tensor
        res = np.zeros((self.dim_batch, self.dim_y))
        print("result's shape", res.shape)

        # create recorder for data movement amount
        wgt_data_mov = 0
        inp_data_mov = 0
        out_data_mov = 0
        # get the stride encoding of the loops to be used in the loops
        stride_code = self.get_dim_stride_code(self.loop_order, self.loop_bounds)

        ## start the hlscnn-ilator systemc simulation
        self.start_simulator()

        for _ in range(self.total_invokes):
            tb_idx = self.get_dim_tile_idx("t", cntr.value, stride_code)
            ty_idx = self.get_dim_tile_idx("y", cntr.value, stride_code)
            tx_idx = self.get_dim_tile_idx("x", cntr.value, stride_code)

            print("current cntr:", cntr.cntr)

            # get the data tile indices
            t_wgt_id = t_wgt_log.index(self._get_tile_encode(self.WGT_DIMS, cntr.value))
            t_act_id = t_act_log.index(self._get_tile_encode(self.INP_DIMS, cntr.value))
            print("tb_idx::tx_idx::ty_idx ", tb_idx, tx_idx, ty_idx)
            print("current tile idx: ", t_wgt_id, t_act_id)

            # determine IO requirements
            is_new_wgt = not (t_wgt_id in wgt_book)
            is_new_act = not (t_act_id in act_book)
            io_info = (is_new_wgt, is_new_act, True)

            # determine the data address
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
            # need to divide the wgt_addr by 4 to get addr for each PE
            addr_info = (wgt_addr >> 2, inp_addr, out_tile_addr)

            # slice the tile data from original data
            t_wgt = self.get_tiled_weight_data(weight, ty_idx, tx_idx)
            t_inp = self.get_tiled_input_data(input, tb_idx, tx_idx)

            # dump the tiled data to files
            if is_new_wgt:
                t_wgt.tofile("./data/wgt.txt", sep="\n")
                wgt_data_mov += t_wgt.size
            if is_new_act:
                t_inp.tofile("./data/inp.txt", sep="\n")
                inp_data_mov += t_inp.size

            layer_info = {
                "num_v_in": t_wgt.shape[1] >> 4,
                "num_v_out": t_wgt.shape[0] >> 4,
                "num_timestep": t_inp.shape[0],
                "is_bias": False,
                "op_name": f"flexasr_linear_layer_{cntr.cntr}",
                "dtype": "float32",
            }
            print("io_info:", io_info)
            print("addr_info:", tuple(map(hex, addr_info)))
            print("layer_info:", layer_info)

            # call the code generation for the tile flexasr linear layer operator
            driver = FlexASRLinearLayerDriver(layer_info, addr_info, io_info)
            tile_result = driver.run()

            assert tile_result is not None
            tile_result = np.array(tile_result).reshape(
                (t_inp.shape[0], t_wgt.shape[0])
            )

            res = self.merge_tiled_result(res, tile_result, tb_idx, ty_idx)
            out_data_mov += tile_result.size

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

def test_scheduler():
    dim_t = 256
    dim_x = 768
    dim_y = 768
    tile_t = 16
    tile_x = 768
    tile_y = 256
    loop_order = "xyt"

    wgt_shape = (dim_y, dim_x)
    inp_shape = (dim_t, dim_x)

    layer_info = (dim_t, dim_x, dim_y)
    schedule = (loop_order, tile_t, tile_x, tile_y)
    test_driver = FlexASRLinearLayerScheduler(layer_info, schedule)

    test_wgt = 0.25 * np.random.uniform(-1, 1, wgt_shape).astype("float32")
    test_inp = 0.25 * np.random.uniform(-1, 1, inp_shape).astype("float32")

    res = test_driver.run(test_wgt, test_inp)
    ref = np.matmul(test_inp, np.transpose(test_wgt))

    print("mismatch: ", cal_single_tensor_error(res, ref))

test_scheduler()
