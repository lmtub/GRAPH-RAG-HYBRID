static void cdrom_pio_impl(int nblocks)

{

    QPCIDevice *dev;

    void *bmdma_base, *ide_base;

    FILE *fh;

    int patt_blocks = MAX(16, nblocks);

    size_t patt_len = ATAPI_BLOCK_SIZE * patt_blocks;

    char *pattern = g_malloc(patt_len);

    size_t rxsize = ATAPI_BLOCK_SIZE * nblocks;

    uint16_t *rx = g_malloc0(rxsize);

    int i, j;

    uint8_t data;

    uint16_t limit;



    /* Prepopulate the CDROM with an interesting pattern */

    generate_pattern(pattern, patt_len, ATAPI_BLOCK_SIZE);

    fh = fopen(tmp_path, "w+");

    fwrite(pattern, ATAPI_BLOCK_SIZE, patt_blocks, fh);

    fclose(fh);



    ide_test_start("-drive if=none,file=%s,media=cdrom,format=raw,id=sr0,index=0 "

                   "-device ide-cd,drive=sr0,bus=ide.0", tmp_path);

    dev = get_pci_device(&bmdma_base, &ide_base);

    qtest_irq_intercept_in(global_qtest, "ioapic");



    /* PACKET command on device 0 */

    qpci_io_writeb(dev, ide_base + reg_device, 0);

    qpci_io_writeb(dev, ide_base + reg_lba_middle, BYTE_COUNT_LIMIT & 0xFF);

    qpci_io_writeb(dev, ide_base + reg_lba_high,

                   (BYTE_COUNT_LIMIT >> 8 & 0xFF));

    qpci_io_writeb(dev, ide_base + reg_command, CMD_PACKET);

    /* HP0: Check_Status_A State */

    nsleep(400);

    data = ide_wait_clear(BSY);

    /* HP1: Send_Packet State */

    assert_bit_set(data, DRQ | DRDY);

    assert_bit_clear(data, ERR | DF | BSY);



    /* SCSI CDB (READ10) -- read n*2048 bytes from block 0 */

    send_scsi_cdb_read10(dev, ide_base, 0, nblocks);



    /* Read data back: occurs in bursts of 'BYTE_COUNT_LIMIT' bytes.

     * If BYTE_COUNT_LIMIT is odd, we transfer BYTE_COUNT_LIMIT - 1 bytes.

     * We allow an odd limit only when the remaining transfer size is

     * less than BYTE_COUNT_LIMIT. However, SCSI's read10 command can only

     * request n blocks, so our request size is always even.

     * For this reason, we assume there is never a hanging byte to fetch. */

    g_assert(!(rxsize & 1));

    limit = BYTE_COUNT_LIMIT & ~1;

    for (i = 0; i < DIV_ROUND_UP(rxsize, limit); i++) {

        size_t offset = i * (limit / 2);

        size_t rem = (rxsize / 2) - offset;



        /* HP3: INTRQ_Wait */

        ide_wait_intr(IDE_PRIMARY_IRQ);



        /* HP2: Check_Status_B (and clear IRQ) */

        data = ide_wait_clear(BSY);

        assert_bit_set(data, DRQ | DRDY);

        assert_bit_clear(data, ERR | DF | BSY);



        /* HP4: Transfer_Data */

        for (j = 0; j < MIN((limit / 2), rem); j++) {

            rx[offset + j] = cpu_to_le16(qpci_io_readw(dev,

                                                       ide_base + reg_data));

        }

    }



    /* Check for final completion IRQ */

    ide_wait_intr(IDE_PRIMARY_IRQ);



    /* Sanity check final state */

    data = ide_wait_clear(DRQ);

    assert_bit_set(data, DRDY);

    assert_bit_clear(data, DRQ | ERR | DF | BSY);



    g_assert_cmpint(memcmp(pattern, rx, rxsize), ==, 0);

    g_free(pattern);

    g_free(rx);

    test_bmdma_teardown();

}
