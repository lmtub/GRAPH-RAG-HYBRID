static uint64_t serial_ioport_read(void *opaque, hwaddr addr, unsigned size)

{

    SerialState *s = opaque;

    uint32_t ret;



    addr &= 7;

    switch(addr) {

    default:

    case 0:

        if (s->lcr & UART_LCR_DLAB) {

            ret = s->divider & 0xff;

        } else {

            if(s->fcr & UART_FCR_FE) {

                ret = fifo8_is_full(&s->recv_fifo) ?

                            0 : fifo8_pop(&s->recv_fifo);

                if (s->recv_fifo.num == 0) {

                    s->lsr &= ~(UART_LSR_DR | UART_LSR_BI);

                } else {

                    qemu_mod_timer(s->fifo_timeout_timer, qemu_get_clock_ns (vm_clock) + s->char_transmit_time * 4);

                }

                s->timeout_ipending = 0;

            } else {

                ret = s->rbr;

                s->lsr &= ~(UART_LSR_DR | UART_LSR_BI);

            }

            serial_update_irq(s);

            if (!(s->mcr & UART_MCR_LOOP)) {

                /* in loopback mode, don't receive any data */

                qemu_chr_accept_input(s->chr);

            }

        }

        break;

    case 1:

        if (s->lcr & UART_LCR_DLAB) {

            ret = (s->divider >> 8) & 0xff;

        } else {

            ret = s->ier;

        }

        break;

    case 2:

        ret = s->iir;

        if ((ret & UART_IIR_ID) == UART_IIR_THRI) {

            s->thr_ipending = 0;

            serial_update_irq(s);

        }

        break;

    case 3:

        ret = s->lcr;

        break;

    case 4:

        ret = s->mcr;

        break;

    case 5:

        ret = s->lsr;

        /* Clear break and overrun interrupts */

        if (s->lsr & (UART_LSR_BI|UART_LSR_OE)) {

            s->lsr &= ~(UART_LSR_BI|UART_LSR_OE);

            serial_update_irq(s);

        }

        break;

    case 6:

        if (s->mcr & UART_MCR_LOOP) {

            /* in loopback, the modem output pins are connected to the

               inputs */

            ret = (s->mcr & 0x0c) << 4;

            ret |= (s->mcr & 0x02) << 3;

            ret |= (s->mcr & 0x01) << 5;

        } else {

            if (s->poll_msl >= 0)

                serial_update_msl(s);

            ret = s->msr;

            /* Clear delta bits & msr int after read, if they were set */

            if (s->msr & UART_MSR_ANY_DELTA) {

                s->msr &= 0xF0;

                serial_update_irq(s);

            }

        }

        break;

    case 7:

        ret = s->scr;

        break;

    }

    DPRINTF("read addr=0x%" HWADDR_PRIx " val=0x%02x\n", addr, ret);

    return ret;

}
