static void test_bmdma_simple_rw(void)

{

    QPCIDevice *dev;

    void *bmdma_base, *ide_base;

    uint8_t status;

    uint8_t *buf;

    uint8_t *cmpbuf;

    size_t len = 512;

    uintptr_t guest_buf = guest_alloc(guest_malloc, len);



    PrdtEntry prdt[] = {

        {

            .addr = cpu_to_le32(guest_buf),

            .size = cpu_to_le32(len | PRDT_EOT),

        },

    };



    dev = get_pci_device(&bmdma_base, &ide_base);



    buf = g_malloc(len);

    cmpbuf = g_malloc(len);



    /* Write 0x55 pattern to sector 0 */

    memset(buf, 0x55, len);

    memwrite(guest_buf, buf, len);



    status = send_dma_request(CMD_WRITE_DMA, 0, 1, prdt,

                              ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, BM_STS_INTR);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);



    /* Write 0xaa pattern to sector 1 */

    memset(buf, 0xaa, len);

    memwrite(guest_buf, buf, len);



    status = send_dma_request(CMD_WRITE_DMA, 1, 1, prdt,

                              ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, BM_STS_INTR);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);



    /* Read and verify 0x55 pattern in sector 0 */

    memset(cmpbuf, 0x55, len);



    status = send_dma_request(CMD_READ_DMA, 0, 1, prdt, ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, BM_STS_INTR);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);



    memread(guest_buf, buf, len);

    g_assert(memcmp(buf, cmpbuf, len) == 0);



    /* Read and verify 0xaa pattern in sector 1 */

    memset(cmpbuf, 0xaa, len);



    status = send_dma_request(CMD_READ_DMA, 1, 1, prdt, ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, BM_STS_INTR);

    assert_bit_clear(qpci_io_readb(dev, ide_base + reg_status), DF | ERR);



    memread(guest_buf, buf, len);

    g_assert(memcmp(buf, cmpbuf, len) == 0);





    g_free(buf);

    g_free(cmpbuf);

}
