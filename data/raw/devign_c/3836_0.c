static gboolean serial_xmit(GIOChannel *chan, GIOCondition cond, void *opaque)

{

    SerialState *s = opaque;



    if (s->tsr_retry <= 0) {

        if (s->fcr & UART_FCR_FE) {

            s->tsr = fifo8_is_full(&s->xmit_fifo) ?

                        0 : fifo8_pop(&s->xmit_fifo);

            if (!s->xmit_fifo.num) {

                s->lsr |= UART_LSR_THRE;

            }

        } else if ((s->lsr & UART_LSR_THRE)) {

            return FALSE;

        } else {

            s->tsr = s->thr;

            s->lsr |= UART_LSR_THRE;

            s->lsr &= ~UART_LSR_TEMT;

        }

    }



    if (s->mcr & UART_MCR_LOOP) {

        /* in loopback mode, say that we just received a char */

        serial_receive1(s, &s->tsr, 1);

    } else if (qemu_chr_fe_write(s->chr, &s->tsr, 1) != 1) {

        if (s->tsr_retry >= 0 && s->tsr_retry < MAX_XMIT_RETRY &&

            qemu_chr_fe_add_watch(s->chr, G_IO_OUT, serial_xmit, s) > 0) {

            s->tsr_retry++;

            return FALSE;

        }

        s->tsr_retry = 0;

    } else {

        s->tsr_retry = 0;

    }



    s->last_xmit_ts = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);



    if (s->lsr & UART_LSR_THRE) {

        s->lsr |= UART_LSR_TEMT;

        s->thr_ipending = 1;

        serial_update_irq(s);

    }



    return FALSE;

}
