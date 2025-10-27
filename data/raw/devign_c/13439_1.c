static uint32_t serial_ioport_read(void *opaque, uint32_t addr)

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

            ret = s->rbr;

            s->lsr &= ~(UART_LSR_DR | UART_LSR_BI);

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

        /* reset THR pending bit */

        if ((ret & 0x7) == UART_IIR_THRI)

            s->thr_ipending = 0;

        serial_update_irq(s);

        break;

    case 3:

        ret = s->lcr;

        break;

    case 4:

        ret = s->mcr;

        break;

    case 5:

        ret = s->lsr;

        break;

    case 6:

        if (s->mcr & UART_MCR_LOOP) {

            /* in loopback, the modem output pins are connected to the

               inputs */

            ret = (s->mcr & 0x0c) << 4;

            ret |= (s->mcr & 0x02) << 3;

            ret |= (s->mcr & 0x01) << 5;

        } else {

            ret = s->msr;

        }

        break;

    case 7:

        ret = s->scr;

        break;

    }

#ifdef DEBUG_SERIAL

    printf("serial: read addr=0x%02x val=0x%02x\n", addr, ret);

#endif

    return ret;

}
