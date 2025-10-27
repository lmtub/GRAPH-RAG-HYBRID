static void serial_ioport_write(void *opaque, uint32_t addr, uint32_t val)

{

    SerialState *s = opaque;

    unsigned char ch;



    addr &= 7;

#ifdef DEBUG_SERIAL

    printf("serial: write addr=0x%02x val=0x%02x\n", addr, val);

#endif

    switch(addr) {

    default:

    case 0:

        if (s->lcr & UART_LCR_DLAB) {

            s->divider = (s->divider & 0xff00) | val;

            serial_update_parameters(s);

        } else {

            s->thr_ipending = 0;

            s->lsr &= ~UART_LSR_THRE;

            serial_update_irq(s);

            ch = val;

            if (!(s->mcr & UART_MCR_LOOP)) {

                /* when not in loopback mode, send the char */

                qemu_chr_write(s->chr, &ch, 1);

            } else {

                /* in loopback mode, say that we just received a char */

                serial_receive_byte(s, ch);

            }

            if (s->tx_burst > 0) {

                s->tx_burst--;

                serial_tx_done(s);

            } else if (s->tx_burst == 0) {

                s->tx_burst--;

                qemu_mod_timer(s->tx_timer, qemu_get_clock(vm_clock) +

                               ticks_per_sec * THROTTLE_TX_INTERVAL / 1000);

            }

        }

        break;

    case 1:

        if (s->lcr & UART_LCR_DLAB) {

            s->divider = (s->divider & 0x00ff) | (val << 8);

            serial_update_parameters(s);

        } else {

            s->ier = val & 0x0f;

            if (s->lsr & UART_LSR_THRE) {

                s->thr_ipending = 1;

            }

            serial_update_irq(s);

        }

        break;

    case 2:

        break;

    case 3:

        {

            int break_enable;

            s->lcr = val;

            serial_update_parameters(s);

            break_enable = (val >> 6) & 1;

            if (break_enable != s->last_break_enable) {

                s->last_break_enable = break_enable;

                qemu_chr_ioctl(s->chr, CHR_IOCTL_SERIAL_SET_BREAK,

                               &break_enable);

            }

        }

        break;

    case 4:

        s->mcr = val & 0x1f;

        break;

    case 5:

        break;

    case 6:

        break;

    case 7:

        s->scr = val;

        break;

    }

}
