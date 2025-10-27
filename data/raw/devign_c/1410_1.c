static int ahci_dma_rw_buf(IDEDMA *dma, int is_write)

{

    AHCIDevice *ad = DO_UPCAST(AHCIDevice, dma, dma);

    IDEState *s = &ad->port.ifs[0];

    uint8_t *p = s->io_buffer + s->io_buffer_index;

    int l = s->io_buffer_size - s->io_buffer_index;



    if (ahci_populate_sglist(ad, &s->sg)) {

        return 0;

    }



    if (is_write) {

        dma_buf_read(p, l, &s->sg);

    } else {

        dma_buf_write(p, l, &s->sg);

    }



    /* update number of transferred bytes */

    ad->cur_cmd->status = cpu_to_le32(le32_to_cpu(ad->cur_cmd->status) + l);

    s->io_buffer_index += l;



    DPRINTF(ad->port_no, "len=%#x\n", l);



    return 1;

}
