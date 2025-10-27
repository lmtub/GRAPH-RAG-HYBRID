static uint64_t exynos4210_uart_read(void *opaque, target_phys_addr_t offset,

                                  unsigned size)

{

    Exynos4210UartState *s = (Exynos4210UartState *)opaque;

    uint32_t res;



    switch (offset) {

    case UERSTAT: /* Read Only */

        res = s->reg[I_(UERSTAT)];

        s->reg[I_(UERSTAT)] = 0;

        return res;

    case UFSTAT: /* Read Only */

        s->reg[I_(UFSTAT)] = fifo_elements_number(&s->rx) & 0xff;

        if (fifo_empty_elements_number(&s->rx) == 0) {

            s->reg[I_(UFSTAT)] |= UFSTAT_Rx_FIFO_FULL;

            s->reg[I_(UFSTAT)] &= ~0xff;

        }

        return s->reg[I_(UFSTAT)];

    case URXH:

        if (s->reg[I_(UFCON)] & UFCON_FIFO_ENABLE) {

            if (fifo_elements_number(&s->rx)) {

                res = fifo_retrieve(&s->rx);

#if DEBUG_Rx_DATA

                fprintf(stderr, "%c", res);

#endif

                if (!fifo_elements_number(&s->rx)) {

                    s->reg[I_(UTRSTAT)] &= ~UTRSTAT_Rx_BUFFER_DATA_READY;

                } else {

                    s->reg[I_(UTRSTAT)] |= UTRSTAT_Rx_BUFFER_DATA_READY;

                }

            } else {

                s->reg[I_(UINTSP)] |= UINTSP_ERROR;

                exynos4210_uart_update_irq(s);

                res = 0;

            }

        } else {

            s->reg[I_(UTRSTAT)] &= ~UTRSTAT_Rx_BUFFER_DATA_READY;

            res = s->reg[I_(URXH)];

        }

        return res;

    case UTXH:

        PRINT_DEBUG("UART%d: Trying to read from WO register: %s [%04x]\n",

                    s->channel, exynos4210_uart_regname(offset), offset);

        break;

    default:

        return s->reg[I_(offset)];

    }



    return 0;

}
