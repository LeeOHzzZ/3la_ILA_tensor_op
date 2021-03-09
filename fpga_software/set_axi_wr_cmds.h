#ifndef SET_AXI_WR_CMDS_H_
#define SET_AXI_WR_CMDS_H_

#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_io.h"
#include "xbasic_types.h"
#include "arm_neon.h"
#include "xscugic.h"
#include "xtime_l.h"
#include "sleep.h"

typedef unsigned uint128_t __attribute__ ((mode (TI)));

#define HW128_REG(ADDRESS)  (*((volatile uint128_t  *)(ADDRESS)))
#define HW64_REG(ADDRESS)  (*((volatile unsigned long long *)(ADDRESS)))
#define HW32_REG(ADDRESS)  (*((volatile unsigned int  *)(ADDRESS)))
#define HW16_REG(ADDRESS)  (*((volatile unsigned short *)(ADDRESS)))
#define HW8_REG(ADDRESS)   (*((volatile unsigned char  *)(ADDRESS)))

typedef union {
  uint128_t val128;
  int64x2_t val64;
  int32x4_t val32;
  int16x8_t val16;
  int8x16_t val8;
} smiv128_t;

smiv128_t weight128;
smiv128_t read_data;

int set_axi_wr_cmds(){

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
    // printf("%s :: %s :: %s \n", addr, data_h, data_l);
    long long addr_int, data_h_int, data_l_int;
    sscanf(addr, "%llx", &addr_int);
    sscanf(data_h, "%llx", &data_h_int);
    sscanf(data_l, "%llx", &data_l_int);

		weight128.val64[1] = data_h_int;
		weight128.val64[0] = data_l_int;
		HW128_REG(addr_int) = weight128.val128;
  }	  
}

#endif /* SET_AXI_WR_CMDS_H_ */
