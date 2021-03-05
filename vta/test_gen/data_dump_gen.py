import sys
import json

def parse_data(data_list):
  ret = []
  for l in data_list:
    if 'output' in l:
      continue
    if 'buf' in l:
      idx_l = l.find('[')
      idx_h = l.find(']')
      if 'input_buffer' in l:
        name = 'input_buffer'
      elif 'weight_buffer' in l:
        name = 'weight_buffer'
      elif 'bias_buf' in l:
        name = 'bias_buffer'
      idx = int(l[idx_l+1 : idx_h])  
      v_idx = l.find(':') + 2
      value = '0x' + l[v_idx:]
      ret.append({
        'name' : name,
        'idx' : idx,
        'value' : value
      })
    if 'uop' in l:
      idx_l = l.find('No.')
      idx_h = l.find('\t')
      idx = int(l[idx_l+3:idx_h-1])
      v_idx = l.find(':') + 2
      value = '0x'+l[v_idx:]
      name = 'uop_buffer'
      ret.append({
        'name' : name,
        'idx' : idx,
        'value' : value
      })
  ret = {'data_dump' : ret}
  return ret

if __name__ == "__main__":
  data_log = sys.argv[1]
  # data_log[data_log.find('/'):]
  dest_path = './data_dump' + data_log[data_log.find('/'):] + '_data_dump.json'

  with open(data_log, 'r') as fin:
    raw_data = fin.read().splitlines()
  
  parsed_data = parse_data(raw_data)
  with open(dest_path, 'w') as fout:
    json.dump(parsed_data, fout, indent=4)
  
  print('data has been dumped to ' + dest_path)
  
  # with open('raw_data_list.json', 'w') as fout:
  #   json.dump(raw_data, fout, indent=4)