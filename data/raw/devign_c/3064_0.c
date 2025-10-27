static void lsi_do_dma(LSIState *s, int out)

{

    uint32_t count;

    target_phys_addr_t addr;



    if (!s->current_dma_len) {

        /* Wait until data is available.  */

        DPRINTF("DMA no data available\n");

        return;

    }



    count = s->dbc;

    if (count > s->current_dma_len)

        count = s->current_dma_len;



    addr = s->dnad;

    if (lsi_dma_40bit(s))

        addr |= ((uint64_t)s->dnad64 << 32);

    else if (s->sbms)

        addr |= ((uint64_t)s->sbms << 32);



    DPRINTF("DMA addr=0x" TARGET_FMT_plx " len=%d\n", addr, count);

    s->csbc += count;

    s->dnad += count;

    s->dbc -= count;



    if (s->dma_buf == NULL) {

        s->dma_buf = s->current_dev->get_buf(s->current_dev,

                                             s->current_tag);

    }



    /* ??? Set SFBR to first data byte.  */

    if (out) {

        cpu_physical_memory_read(addr, s->dma_buf, count);

    } else {

        cpu_physical_memory_write(addr, s->dma_buf, count);

    }

    s->current_dma_len -= count;

    if (s->current_dma_len == 0) {

        s->dma_buf = NULL;

        if (out) {

            /* Write the data.  */

            s->current_dev->write_data(s->current_dev, s->current_tag);

        } else {

            /* Request any remaining data.  */

            s->current_dev->read_data(s->current_dev, s->current_tag);

        }

    } else {

        s->dma_buf += count;

        lsi_resume_script(s);

    }

}
