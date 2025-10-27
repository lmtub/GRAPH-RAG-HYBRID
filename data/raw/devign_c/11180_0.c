uint64_t mcf_uart_read(void *opaque, target_phys_addr_t addr,

                       unsigned size)

{

    mcf_uart_state *s = (mcf_uart_state *)opaque;

    switch (addr & 0x3f) {

    case 0x00:

        return s->mr[s->current_mr];

    case 0x04:

        return s->sr;

    case 0x0c:

        {

            uint8_t val;

            int i;



            if (s->fifo_len == 0)

                return 0;



            val = s->fifo[0];

            s->fifo_len--;

            for (i = 0; i < s->fifo_len; i++)

                s->fifo[i] = s->fifo[i + 1];

            s->sr &= ~MCF_UART_FFULL;

            if (s->fifo_len == 0)

                s->sr &= ~MCF_UART_RxRDY;

            mcf_uart_update(s);

            qemu_chr_accept_input(s->chr);

            return val;

        }

    case 0x10:

        /* TODO: Implement IPCR.  */

        return 0;

    case 0x14:

        return s->isr;

    case 0x18:

        return s->bg1;

    case 0x1c:

        return s->bg2;

    default:

        return 0;

    }

}
