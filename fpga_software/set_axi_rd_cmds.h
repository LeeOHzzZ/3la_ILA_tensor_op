#ifndef SET_AXI_RD_CMDS_H_
#define SET_AXI_RD_CMDS_H_

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

  FILE *fin, *fout;
  fin = fopen("./axi_rd_cmds.txt", "r");
  if (fin == NULL) {
    perror("fopen for read data");
    return -1;
  }
  fout = fopen("./flexnlp_result.txt", "w");
  if (fout == NULL) {
    perror("fopen for write result");
    return -1;
  }

  printf("%s\n", "opened file axi_rd_cmds.txt, and write results to flexnlp_result.txt");
  char *addr = NULL;
  char *read_line = NULL;
  size_t len;
  __ssize_t line_size;
  while (line_size = getline(&read_line, &len, fin) >= 0) {
    // printf("\n%s\n", data);
    addr = strtok(read_line, ";");
    long long addr_int;
    sscanf(addr, "%llx", &addr_int);
    read_data.val128 = HW128_REG(addr);
    fprintf(fout, "0x%llx;0x%016llx%016llx\n", addr, read_data.val64[1], read_data.val64[0]);
  }
  fclose(fout);
}

#endif /* SET_AXI_RD_CMDS_H_ */
