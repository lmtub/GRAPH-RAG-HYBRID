static void test_bmdma_no_busmaster(void)

{

    QPCIDevice *dev;

    void *bmdma_base, *ide_base;

    uint8_t status;



    dev = get_pci_device(&bmdma_base, &ide_base);



    /* No PRDT_EOT, each entry addr 0/size 64k, and in theory qemu shouldn't be

     * able to access it anyway because the Bus Master bit in the PCI command

     * register isn't set. This is complete nonsense, but it used to be pretty

     * good at confusing and occasionally crashing qemu. */

    PrdtEntry prdt[4096] = { };



    status = send_dma_request(CMD_READ_DMA | CMDF_NO_BM, 0, 512,

                              prdt, ARRAY_SIZE(prdt), NULL);



    /* Not entirely clear what the expected result is, but this is what we get

     * in practice. At least we want to be aware of any changes. */

    g_assert_cmphex(status, ==, BM_STS_ACTIVE | BM_STS_INTR);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);

}
