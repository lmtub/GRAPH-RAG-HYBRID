static void ahci_start_transfer(IDEDMA *dma)

{

    AHCIDevice *ad = DO_UPCAST(AHCIDevice, dma, dma);

    IDEState *s = &ad->port.ifs[0];

    uint32_t size = (uint32_t)(s->data_end - s->data_ptr);

    /* write == ram -> device */

    uint16_t opts = le16_to_cpu(ad->cur_cmd->opts);

    int is_write = opts & AHCI_CMD_WRITE;

    int is_atapi = opts & AHCI_CMD_ATAPI;

    int has_sglist = 0;



    if (is_atapi && !ad->done_atapi_packet) {

        /* already prepopulated iobuffer */

        ad->done_atapi_packet = true;

        size = 0;

        goto out;

    }



    if (ahci_dma_prepare_buf(dma, is_write)) {

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



out:

    /* declare that we processed everything */

    s->data_ptr = s->data_end;



    /* Update number of transferred bytes, destroy sglist */

    ahci_commit_buf(dma, size);



    s->end_transfer_func(s);



    if (!(s->status & DRQ_STAT)) {

        /* done with PIO send/receive */

        ahci_write_fis_pio(ad, le32_to_cpu(ad->cur_cmd->status));

    }

}
