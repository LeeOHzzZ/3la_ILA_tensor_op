import numpy as np
from math import sqrt

def cal_error(result, ref):
  mean_diff = np.mean(np.abs(result - ref))
  return mean_diff/np.mean(np.abs(result)), mean_diff/np.mean(np.abs(ref))


def cal_mean_stdd(data_list):
  """
  This function calculate the mean and standard deviation of the input data list
  """
  mean = sum(data_list) / len(data_list)
  stdd = sqrt(sum(list(map(lambda x: (x - mean)**2, data_list)))/len(data_list))
  return mean, stdd


for size in range(10, 500, 10):
  err_list = []
  for i in range(100):
    x = np.random.uniform(-1, 1, (size, size))
    y = np.random.uniform(-1, 1, (size, size))
    err, _ = cal_error(x, y)
    err_list.append(err)
  mean, stdd = cal_mean_stdd(err_list)
  print(f"tensor size: ({size}, {size}): mean err: {mean:.5%}; stdd: {stdd:.5%};")

