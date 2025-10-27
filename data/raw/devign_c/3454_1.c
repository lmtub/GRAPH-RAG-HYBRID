static void rocker_test_dma_ctrl(Rocker *r, uint32_t val)

{

    PCIDevice *dev = PCI_DEVICE(r);

    char *buf;

    int i;



    buf = g_malloc(r->test_dma_size);



    if (!buf) {

        DPRINTF("test dma buffer alloc failed");

        return;

    }



    switch (val) {

    case ROCKER_TEST_DMA_CTRL_CLEAR:

        memset(buf, 0, r->test_dma_size);

        break;

    case ROCKER_TEST_DMA_CTRL_FILL:

        memset(buf, 0x96, r->test_dma_size);

        break;

    case ROCKER_TEST_DMA_CTRL_INVERT:

        pci_dma_read(dev, r->test_dma_addr, buf, r->test_dma_size);

        for (i = 0; i < r->test_dma_size; i++) {

            buf[i] = ~buf[i];

        }

        break;

    default:

        DPRINTF("not test dma control val=0x%08x\n", val);

        goto err_out;

    }

    pci_dma_write(dev, r->test_dma_addr, buf, r->test_dma_size);



    rocker_msix_irq(r, ROCKER_MSIX_VEC_TEST);



err_out:

    g_free(buf);

}
