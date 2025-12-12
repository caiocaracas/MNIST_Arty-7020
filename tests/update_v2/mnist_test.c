#include "xparameters.h"
#include "xaxidma.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xstatus.h"

#include <stdint.h>


#define MNIST_ACCEL_BASE_ADDR     XPAR_MNIST_ACCEL_0_BASEADDR

#define MNIST_REG_CTRL_OFFSET     0x00U
#define MNIST_REG_STATUS_OFFSET   0x04U
#define MNIST_REG_IMG_LEN_OFFSET  0x08U

#define MNIST_CTRL_START_MASK     0x00000001U

#define MNIST_STATUS_DONE_MASK    0x00000001U
#define MNIST_STATUS_ERROR_MASK   0x00000004U

#define DMA_DEV_ID                XPAR_XAXIDMA_0_DEVICE_ID

#define DMA_TO_DEV                XAXIDMA_DMA_TO_DEVICE
#define DEV_TO_DMA                XAXIDMA_DEVICE_TO_DMA

#define MNIST_IMG_BYTES           784U
#define MNIST_NUM_LOGITS          10U
#define MNIST_LOGITS_BYTES        16U
#define DMA_BUF_ALIGNMENT         64U
#define NUM_TEST_IMAGES           10U

static XAxiDma AxiDma;

static uint8_t ImageBuf[MNIST_IMG_BYTES]
    __attribute__((aligned(DMA_BUF_ALIGNMENT)));

static uint8_t LogitsBuf[MNIST_LOGITS_BYTES]
    __attribute__((aligned(DMA_BUF_ALIGNMENT)));

extern const int8_t mnist_images[NUM_TEST_IMAGES][MNIST_IMG_BYTES];
extern const uint8_t mnist_labels[NUM_TEST_IMAGES];

static int InitDma(void)
{
    XAxiDma_Config *CfgPtr;
    int Status;

    CfgPtr = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!CfgPtr) {
        xil_printf("DMA lookup failed\r\n");
        return XST_FAILURE;
    }

    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("DMA init failed\r\n");
        return Status;
    }

    if (XAxiDma_HasSg(&AxiDma)) {
        xil_printf("DMA in SG mode, not supported\r\n");
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}

static int RunInference(const int8_t *image)
{
    uint32_t status;

    /* copy image to DMA buffer */
    for (uint32_t i = 0; i < MNIST_IMG_BYTES; i++) {
        ImageBuf[i] = (uint8_t)image[i];
    }

    Xil_DCacheFlushRange((UINTPTR)ImageBuf, MNIST_IMG_BYTES);
    Xil_DCacheInvalidateRange((UINTPTR)LogitsBuf, MNIST_LOGITS_BYTES);

    Xil_Out32(MNIST_ACCEL_BASE_ADDR + MNIST_REG_IMG_LEN_OFFSET,
              MNIST_IMG_BYTES);

    XAxiDma_SimpleTransfer(&AxiDma,
                           (UINTPTR)LogitsBuf,
                           MNIST_LOGITS_BYTES,
                           DEV_TO_DMA);

    XAxiDma_SimpleTransfer(&AxiDma,
                           (UINTPTR)ImageBuf,
                           MNIST_IMG_BYTES,
                           DMA_TO_DEV);

    Xil_Out32(MNIST_ACCEL_BASE_ADDR + MNIST_REG_CTRL_OFFSET,
              MNIST_CTRL_START_MASK);

    do {
        status = Xil_In32(MNIST_ACCEL_BASE_ADDR + MNIST_REG_STATUS_OFFSET);
    } while ((status & MNIST_STATUS_DONE_MASK) == 0);

    if (status & MNIST_STATUS_ERROR_MASK) {
        xil_printf("Accelerator error\r\n");
        return XST_FAILURE;
    }

    while (XAxiDma_Busy(&AxiDma, DMA_TO_DEV));
    while (XAxiDma_Busy(&AxiDma, DEV_TO_DMA));

    Xil_DCacheInvalidateRange((UINTPTR)LogitsBuf, MNIST_LOGITS_BYTES);
    return XST_SUCCESS;
}

static uint8_t ArgMax(const uint8_t *logits)
{
    int8_t max_val = -128;
    uint8_t max_idx = 0;

    for (uint8_t i = 0; i < MNIST_NUM_LOGITS; i++) {
        int8_t v = (int8_t)logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return max_idx;
}
int main(void)
{
    int correct = 0;

    if (InitDma() != XST_SUCCESS) {
        xil_printf("DMA init failed\r\n");
        return XST_FAILURE;
    }
    for (uint32_t img = 0; img < NUM_TEST_IMAGES; img++) {

        xil_printf("\r\nImage %lu\r\n", img);

        if (RunInference(mnist_images[img]) != XST_SUCCESS) {
            xil_printf("Inference failed\r\n");
            return XST_FAILURE;
        }
        for (uint8_t i = 0; i < MNIST_NUM_LOGITS; i++) {
            xil_printf("%d ", (int8_t)LogitsBuf[i]);
        }
        xil_printf("\r\n");

        uint8_t pred = ArgMax(LogitsBuf);
        uint8_t exp  = mnist_labels[img];

        xil_printf("Predicted: %d\r\n", pred);
        xil_printf("Expected: %d\r\n", exp);

        if (pred == exp) {
            correct++;
        }
    }
    xil_printf("\r\nFINAL RESULT: %d correct out of %d\r\n",
               correct, NUM_TEST_IMAGES);
    xil_printf("Accuracy: %d%%\r\n",
               (correct * 100) / NUM_TEST_IMAGES);
    return XST_SUCCESS;
}
