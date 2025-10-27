static int ahci_start_transfer(IDEDMA *dma)

{

    AHCIDevice *ad = DO_UPCAST(AHCIDevice, dma, dma);

    IDEState *s = &ad->port.ifs[0];

    uint32_t size = (uint32_t)(s->data_end - s->data_ptr);

    /* write == ram -> device */

    uint32_t opts = le32_to_cpu(ad->cur_cmd->opts);

    int is_write = opts & AHCI_CMD_WRITE;

    int is_atapi = opts & AHCI_CMD_ATAPI;

    int has_sglist = 0;



    if (is_atapi && !ad->done_atapi_packet) {

        /* already prepopulated iobuffer */

        ad->done_atapi_packet = 1;

        goto out;

    }



    if (!ahci_populate_sglist(ad, &s->sg)) {

        has_sglist = 1;

    }



    DPRINTF(ad->port_no, "%sing %d bytes on %s w/%s sglist\n",

            is_write ? "writ" : "read", size, is_atapi ? "atapi" : "ata",

            has_sglist ? "" : "o");



    if (has_sglist && size) {

        if (is_write) {

            dma_buf_write(s->data_ptr, size, &s->sg);

        } else {

            dma_buf_read(s->data_ptr, size, &s->sg);

        }

    }



    /* update number of transferred bytes */

    ad->cur_cmd->status = cpu_to_le32(le32_to_cpu(ad->cur_cmd->status) + size);



out:

    /* declare that we processed everything */

    s->data_ptr = s->data_end;



    if (has_sglist) {

        qemu_sglist_destroy(&s->sg);

    }



    s->end_transfer_func(s);



    if (!(s->status & DRQ_STAT)) {

        /* done with DMA */

        ahci_trigger_irq(ad->hba, ad, PORT_IRQ_STAT_DSS);

    }



    return 0;

}
