static void lsi_do_dma(LSIState *s, int out)

{

    uint32_t count, id;

    target_phys_addr_t addr;

    SCSIDevice *dev;



    assert(s->current);

    if (!s->current->dma_len) {

        /* Wait until data is available.  */

        DPRINTF("DMA no data available\n");

        return;

    }



    id = (s->current->tag >> 8) & 0xf;

    dev = s->bus.devs[id];

    if (!dev) {

        lsi_bad_selection(s, id);

        return;

    }



    count = s->dbc;

    if (count > s->current->dma_len)

        count = s->current->dma_len;



    addr = s->dnad;

    /* both 40 and Table Indirect 64-bit DMAs store upper bits in dnad64 */

    if (lsi_dma_40bit(s) || lsi_dma_ti64bit(s))

        addr |= ((uint64_t)s->dnad64 << 32);

    else if (s->dbms)

        addr |= ((uint64_t)s->dbms << 32);

    else if (s->sbms)

        addr |= ((uint64_t)s->sbms << 32);



    DPRINTF("DMA addr=0x" TARGET_FMT_plx " len=%d\n", addr, count);

    s->csbc += count;

    s->dnad += count;

    s->dbc -= count;



    if (s->current->dma_buf == NULL) {

        s->current->dma_buf = dev->info->get_buf(dev, s->current->tag);

    }



    /* ??? Set SFBR to first data byte.  */

    if (out) {

        cpu_physical_memory_read(addr, s->current->dma_buf, count);

    } else {

        cpu_physical_memory_write(addr, s->current->dma_buf, count);

    }

    s->current->dma_len -= count;

    if (s->current->dma_len == 0) {

        s->current->dma_buf = NULL;

        if (out) {

            /* Write the data.  */

            dev->info->write_data(dev, s->current->tag);

        } else {

            /* Request any remaining data.  */

            dev->info->read_data(dev, s->current->tag);

        }

    } else {

        s->current->dma_buf += count;

        lsi_resume_script(s);

    }

}
