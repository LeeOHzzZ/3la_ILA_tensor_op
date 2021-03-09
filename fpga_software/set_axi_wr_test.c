#ifndef SET_AXI_WR_CMDS_H_
#define SET_AXI_WR_CMDS_H_

#include <stdio.h>
#include <string.h>

int main() {
  // std::ifstream fin;
  FILE *fin;
  fin = fopen("./axi_wr_cmds.txt", "r");
  if (fin == NULL) {
    perror("fopen");
    return -1;
  }
  printf("%s\n", "opened file axi_wr_cmds.txt");
  char *addr = NULL, *data_h = NULL, *data_l = NULL;
  char *data = NULL;
  size_t len;
  __ssize_t line_size;
  while (line_size = getline(&data, &len, fin) >= 0) {
    // printf("\n%s\n", data);
    addr = strtok(data, ";");
    data_h = strtok(NULL, ";");
    data_l = strtok(NULL, ";");
    printf("%s :: %s :: %s \n", addr, data_h, data_l);
    long long x, y, z;
    sscanf(addr, "%llx", &x);
    sscanf(data_h, "%llx", &y);
    sscanf(data_l, "%llx", &z);
    printf("%llx :: %llx :: %llx \n", x, y, z);
  }
}

#endif