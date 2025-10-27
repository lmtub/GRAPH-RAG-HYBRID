static void test_bmdma_one_sector_short_prdt(void)

{

    QPCIDevice *dev;

    void *bmdma_base, *ide_base;

    uint8_t status;



    /* Read 2 sectors but only give 1 sector in PRDT */

    PrdtEntry prdt[] = {

        {

            .addr = 0,

            .size = cpu_to_le32(0x200 | PRDT_EOT),

        },

    };



    dev = get_pci_device(&bmdma_base, &ide_base);



    /* Normal request */

    status = send_dma_request(CMD_READ_DMA, 0, 2,

                              prdt, ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, 0);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);



    /* Abort the request before it completes */

    status = send_dma_request(CMD_READ_DMA | CMDF_ABORT, 0, 2,

                              prdt, ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, 0);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);

}
