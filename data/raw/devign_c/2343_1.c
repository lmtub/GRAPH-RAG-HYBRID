static uint32_t get_cmd(ESPState *s, uint8_t *buf)

{

    uint32_t dmalen;

    int target;



    dmalen = s->rregs[ESP_TCLO] | (s->rregs[ESP_TCMID] << 8);

    target = s->wregs[ESP_WBUSID] & 7;

    DPRINTF("get_cmd: len %d target %d\n", dmalen, target);

    if (s->dma) {

        s->dma_memory_read(s->dma_opaque, buf, dmalen);

    } else {

        buf[0] = 0;

        memcpy(&buf[1], s->ti_buf, dmalen);

        dmalen++;

    }



    s->ti_size = 0;

    s->ti_rptr = 0;

    s->ti_wptr = 0;



    if (s->current_dev) {

        /* Started a new command before the old one finished.  Cancel it.  */

        s->current_dev->cancel_io(s->current_dev, 0);

        s->async_len = 0;

    }



    if (target >= ESP_MAX_DEVS || !s->scsi_dev[target]) {

        // No such drive

        s->rregs[ESP_RSTAT] = 0;

        s->rregs[ESP_RINTR] = INTR_DC;

        s->rregs[ESP_RSEQ] = SEQ_0;

        esp_raise_irq(s);

        return 0;

    }

    s->current_dev = s->scsi_dev[target];

    return dmalen;

}
