import os
import json
import numpy as np
import matplotlib.pyplot as plt

rnn_iteration = 10

def get_fname(flist, target, size: str):
    res = list(filter(lambda x: target in x and size in x and "csv" in x, flist))
    assert len(res) == 1, f"More than one files with the target file name; res_list: {res}"
    return res[0]


def gen_summary():
    op_name = [
        "attention",
        "layernorm",
        "linear_layer",
        "lstm",
        "maxpooling",
        "meanpooling"
    ]

    size_list = [64, 128 ,256, 512]

    summary = {}
    # read files for a given operator
    for op in op_name:
        summary[op] = {}
        flist = os.listdir(op)
        # separate size list
        for size in size_list:
            summary[op][str(size)] = {}
            cur_log = summary[op][str(size)]
            avg_mm_list = np.fromfile(
                f"{op}/{get_fname(flist, 'avg_mm', str(size))}",
                sep='\n'
            )
            ts_stdd_list = np.fromfile(
                f"{op}/{get_fname(flist, 'stdd', str(size))}",
                sep='\n'
            )
            print(f"{op} - avg_mm_list tensor shape: {avg_mm_list.shape}")
            print(f"{op} - ts_stdd_list tensor shape: {ts_stdd_list.shape}")
            if op == "lstm":
                # separate each iteration of RNN
                avg_mm_list = np.transpose(avg_mm_list.reshape((-1, rnn_iteration)))
                ts_stdd_list = np.transpose(ts_stdd_list.reshape((-1, rnn_iteration)))
                for i in range(rnn_iteration):
                    cur_avg_mm_list = avg_mm_list[i]
                    cur_ts_stdd_list = ts_stdd_list[i]
                    cur_log[f"No.{i}"] = {
                        # "avg_ts_stdd": f"{np.std(cur_ts_stdd_list):.5%}",
                        "avg_mismatch": f"{np.mean(cur_avg_mm_list):.5%}",
                        "avg_mismatch_stdd": f"{np.std(cur_avg_mm_list):.5%}"
                    }
            else:
                # cur_log["avg_ts_stdd"] = f"{np.std(ts_stdd_list):.5%}"
                cur_log["avg_mismatch"] = f"{np.mean(avg_mm_list):.5%}"
                cur_log["avg_mismatch_stdd"] = f"{np.std(avg_mm_list):.5%}"

    with open("summary.json", "w") as fout:
        json.dump(summary, fout, indent=4)
    
    print("result has been written to summary.json")



def plot():
    # boxplot example 
    data = np.fromfile("op_sim/attention/avg_mm_128.csv", sep='\n')
    fig = plt.figure(figsize=(10, 10))

    plt.boxplot(data)
    plt.savefig(fname="op_sim/attention/avg_mm_128.png", format="png")


if __name__ == "__main__":
    gen_summary()