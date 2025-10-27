static void esp_pci_dma_memory_rw(PCIESPState *pci, uint8_t *buf, int len,

                                  DMADirection dir)

{

    dma_addr_t addr;

    DMADirection expected_dir;



    if (pci->dma_regs[DMA_CMD] & DMA_CMD_DIR) {

        expected_dir = DMA_DIRECTION_FROM_DEVICE;

    } else {

        expected_dir = DMA_DIRECTION_TO_DEVICE;

    }



    if (dir != expected_dir) {

        trace_esp_pci_error_invalid_dma_direction();

        return;

    }



    if (pci->dma_regs[DMA_STAT] & DMA_CMD_MDL) {

        qemu_log_mask(LOG_UNIMP, "am53c974: MDL transfer not implemented\n");

    }



    addr = pci->dma_regs[DMA_SPA];

    if (pci->dma_regs[DMA_WBC] < len) {

        len = pci->dma_regs[DMA_WBC];

    }



    pci_dma_rw(PCI_DEVICE(pci), addr, buf, len, dir);



    /* update status registers */

    pci->dma_regs[DMA_WBC] -= len;

    pci->dma_regs[DMA_WAC] += len;



}