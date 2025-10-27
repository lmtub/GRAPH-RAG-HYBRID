static void serial_xmit(SerialState *s)

{

    do {

        assert(!(s->lsr & UART_LSR_TEMT));

        if (s->tsr_retry == 0) {

            assert(!(s->lsr & UART_LSR_THRE));



            if (s->fcr & UART_FCR_FE) {

                assert(!fifo8_is_empty(&s->xmit_fifo));

                s->tsr = fifo8_pop(&s->xmit_fifo);

                if (!s->xmit_fifo.num) {

                    s->lsr |= UART_LSR_THRE;

                }

            } else {

                s->tsr = s->thr;

                s->lsr |= UART_LSR_THRE;

            }

            if ((s->lsr & UART_LSR_THRE) && !s->thr_ipending) {

                s->thr_ipending = 1;

                serial_update_irq(s);

            }

        }



        if (s->mcr & UART_MCR_LOOP) {

            /* in loopback mode, say that we just received a char */

            serial_receive1(s, &s->tsr, 1);

        } else if (qemu_chr_fe_write(s->chr, &s->tsr, 1) != 1) {

            if (s->tsr_retry < MAX_XMIT_RETRY &&

                qemu_chr_fe_add_watch(s->chr, G_IO_OUT|G_IO_HUP,

                                      serial_watch_cb, s) > 0) {

                s->tsr_retry++;

                return;

            }

        }

        s->tsr_retry = 0;



        /* Transmit another byte if it is already available. It is only

           possible when FIFO is enabled and not empty. */

    } while (!(s->lsr & UART_LSR_THRE));



    s->last_xmit_ts = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    s->lsr |= UART_LSR_TEMT;

}
