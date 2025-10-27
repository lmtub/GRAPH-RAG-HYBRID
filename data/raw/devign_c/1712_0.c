static void stream_process_mem2s(struct Stream *s,

                                 StreamSlave *tx_dev)

{

    uint32_t prev_d;

    unsigned char txbuf[16 * 1024];

    unsigned int txlen;

    uint32_t app[6];



    if (!stream_running(s) || stream_idle(s)) {

        return;

    }



    while (1) {

        stream_desc_load(s, s->regs[R_CURDESC]);



        if (s->desc.status & SDESC_STATUS_COMPLETE) {

            s->regs[R_DMASR] |= DMASR_HALTED;

            break;

        }



        if (stream_desc_sof(&s->desc)) {

            s->pos = 0;

            memcpy(app, s->desc.app, sizeof app);

        }



        txlen = s->desc.control & SDESC_CTRL_LEN_MASK;

        if ((txlen + s->pos) > sizeof txbuf) {

            hw_error("%s: too small internal txbuf! %d\n", __func__,

                     txlen + s->pos);

        }



        cpu_physical_memory_read(s->desc.buffer_address,

                                 txbuf + s->pos, txlen);

        s->pos += txlen;



        if (stream_desc_eof(&s->desc)) {

            stream_push(tx_dev, txbuf, s->pos, app);

            s->pos = 0;

            stream_complete(s);

        }



        /* Update the descriptor.  */

        s->desc.status = txlen | SDESC_STATUS_COMPLETE;

        stream_desc_store(s, s->regs[R_CURDESC]);



        /* Advance.  */

        prev_d = s->regs[R_CURDESC];

        s->regs[R_CURDESC] = s->desc.nxtdesc;

        if (prev_d == s->regs[R_TAILDESC]) {

            s->regs[R_DMASR] |= DMASR_IDLE;

            break;

        }

    }

}
