#include "xparameters.h"
#include "xil_io.h"
#include "xbasic_types.h"
#include "set_axi_wr_cmds.h"
#include "set_axi_rd_cmds.h"
#include "arm_neon.h"
#include "xscugic.h"
#include "xtime_l.h"
#include "sleep.h"

XScuGic IntcInstance;
static void flexnlpISR();
XTime tIrq;

int main() {
    int errors = 0;

    u32 status;
    XScuGic_Config *IntcConfig;


    IntcConfig = XScuGic_LookupConfig(XPAR_PSU_ACPU_GIC_DEVICE_ID);
    status = XScuGic_CfgInitialize(&IntcInstance, IntcConfig, IntcConfig->CpuBaseAddress);

    if(status != XST_SUCCESS){
    	xil_printf("Interrupt controller initialization failed...");
    	return -1;
    }

    XScuGic_SetPriorityTriggerType(&IntcInstance, XPAR_FABRIC_MYFLEXASR_IP_0_INTR_INTR, 0xA0, 3);
    status = XScuGic_Connect(&IntcInstance, XPAR_FABRIC_MYFLEXASR_IP_0_INTR_INTR, (Xil_InterruptHandler)flexnlpISR,0);

    if(status != XST_SUCCESS){
    	xil_printf("Interrupt connection failed...");
    	return -1;
    }

    XScuGic_Enable(&IntcInstance, XPAR_FABRIC_MYFLEXASR_IP_0_INTR_INTR);

    //Linking GIC with the OS
    Xil_ExceptionInit();
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT, (Xil_ExceptionHandler)XScuGic_InterruptHandler, (void *)&IntcInstance);
    Xil_ExceptionEnable();


	xil_printf("************Start of FlexNLP IP test******************\n\r");

    set_axi_wr_cmds();
    sleep(5);
    // TODO: insert the wait for interrupt signal here
    set_axi_rd_cmds();
	// read_data.val128 = HW128_REG(0xA0500000);
	// printf("Read data in main is: %016llx %016llx\n", read_data.val64[1], read_data.val64[0]);

	return 0;
}

static void flexnlpISR() {
	XScuGic_Disable(&IntcInstance, XPAR_FABRIC_MYFLEXASR_IP_0_INTR_INTR);
	xil_printf("Received Interrupt \n");
	XTime_GetTime(&tIrq);
	printf ("IRQ raised at %llu clock cycles.\n", tIrq);
	XScuGic_Enable(&IntcInstance, XPAR_FABRIC_MYFLEXASR_IP_0_INTR_INTR);
}
