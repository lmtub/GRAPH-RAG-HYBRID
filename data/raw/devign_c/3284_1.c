static void make_dirty(uint8_t device)

{

    QPCIDevice *dev;

    QPCIBar bmdma_bar, ide_bar;

    uint8_t status;

    size_t len = 512;

    uintptr_t guest_buf;

    void* buf;



    dev = get_pci_device(&bmdma_bar, &ide_bar);



    guest_buf = guest_alloc(guest_malloc, len);

    buf = g_malloc(len);

    memset(buf, rand() % 255 + 1, len);

    g_assert(guest_buf);

    g_assert(buf);



    memwrite(guest_buf, buf, len);



    PrdtEntry prdt[] = {

        {

            .addr = cpu_to_le32(guest_buf),

            .size = cpu_to_le32(len | PRDT_EOT),

        },

    };



    status = send_dma_request(CMD_WRITE_DMA, 1, 1, prdt,

                              ARRAY_SIZE(prdt), NULL);

    g_assert_cmphex(status, ==, BM_STS_INTR);

    assert_bit_clear(qpci_io_readb(dev, ide_bar, reg_status), DF | ERR);



    g_free(buf);


}