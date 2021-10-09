import sys
import os
import subprocess
import numpy as np


from linear_layer_driver import linear_layer_driver
from lstm_driver import lstm_layer_driver
from pooling_driver import pooling_layer_driver
from layernorm_driver import layernorm_driver
from attention_driver import attention_layer
from src.utils import tool

iteration = 200
def test_linear_layer():
    subprocess.run(["mkdir", "-p", "op_sim/linear_layer"])
    v_size_list = [32, 16, 8, 4]
    for v in v_size_list:
        test_driver = linear_layer_driver(
            num_v_in=v,
            num_v_out=v,
            num_timestep=iteration,
            is_bias=1,
            dtype="float32"
        )
        err_list, ts_stdd_list = test_driver.run_test()
        np.array(err_list).tofile(
            "./op_sim/linear_layer/" + f"avg_mm_{v*16}x{v*16}.csv",
            sep='\n'
        )
        np.array(err_list).tofile(
            "./op_sim/linear_layer/" + f"ts_stdd_{v*16}x{v*16}.csv",
            sep='\n'
        )


def test_maxpooling():
    subprocess.run(["mkdir", "-p", "op_sim/maxpooling"])
    v_size_list = [32, 16, 8, 4]
    for v in v_size_list:
        test_driver = pooling_layer_driver(
            mode = "max",
            num_v_in=v,
            num_ts=iteration*2
        )
        err_list, ts_stdd_list = test_driver.run_test()
        np.array(err_list).tofile(
            "./op_sim/maxpooling/" + f"avg_mm_{v*16}x2.csv",
            sep='\n'
        )
        np.array(ts_stdd_list).tofile(
            "./op_sim/maxpooling/" + f"ts_stdd_{v*16}x2.csv",
            sep='\n'
        )


def test_meanpooling():
    subprocess.run(["mkdir", "-p", "op_sim/meanpooling"])
    v_size_list = [32, 16, 8, 4]
    for v in v_size_list:
        test_driver = pooling_layer_driver(
            mode = "mean",
            num_v_in=v,
            num_ts=iteration*2
        )
        err_list, ts_stdd_list = test_driver.run_test()
        np.array(err_list).tofile(
            "./op_sim/meanpooling/" + f"avg_mm_{v*16}x2.csv",
            sep='\n'
        )
        np.array(ts_stdd_list).tofile(
            "./op_sim/meanpooling/" + f"ts_stdd_{v*16}x2.csv",
            sep='\n'
        )


def test_layernorm():
    subprocess.run(["mkdir", "-p", "op_sim/layernorm"])
    v_size_list = [32, 16, 8, 4]
    for v in v_size_list:
        test_driver = layernorm_driver(
            num_v=v,
            num_ts=iteration
        )
        err_list, ts_stdd_list = test_driver.run_test()
        np.array(err_list).tofile(
            "./op_sim/layernorm/" + f"avg_mm_{v*16}.csv",
            sep='\n'
        )
        np.array(ts_stdd_list).tofile(
            "./op_sim/layernorm/" + f"ts_stdd_{v*16}.csv",
            sep='\n'
        )


def test_attention():
    subprocess.run(["mkdir", "-p", "op_sim/attention"])
    v_size_list = [32, 16, 8, 4]
    for v in v_size_list:
        err_list = []
        stdd_list = []
        for i in range(iteration):
            test_driver = attention_layer(
                num_v=v,
                num_ts=10,
                mem_idx_dec=0,
                mem_idx_enc=0,
            )
            err, stdd = test_driver.run_test()
            err_list.append(err)
            stdd_list.append(stdd)
        np.array(err_list).tofile(
            "./op_sim/attention/" + f"avg_mm_{v*16}.csv",
            sep='\n'
        )
        np.array(stdd_list).tofile(
            "./op_sim/attention/" + f"ts_stdd_{v*16}.csv",
            sep='\n'
        )



def test_lstm():
    subprocess.run(["mkdir", "-p", "op_sim/lstm"])
    v_size_list = [32, 16, 8, 4]
    for v in v_size_list:
        err_list = []
        stdd_list = []
        for i in range(iteration):
            test_driver = lstm_layer_driver(
                num_v_in=v,
                num_v_out=v,
                num_ts=10,
                is_bias=1,
                is_zero_first=1
            )
            cur_err_list, cur_ts_stdd_list = test_driver.run_test(use_relay=0)
            err_list += cur_err_list
            stdd_list += cur_ts_stdd_list
        np.array(err_list).tofile(
            "./op_sim/lstm/" + f"avg_mm_{v*16}x{v*16}.csv",
            sep='\n'
        )
        np.array(stdd_list).tofile(
            "./op_sim/lstm/" + f"ts_stdd_{v*16}x{v*16}.csv",
            sep='\n'
        )


if __name__ == "__main__":
    test_maxpooling()
    test_meanpooling()
    test_layernorm()
    test_attention()
    test_linear_layer()
    test_lstm()

