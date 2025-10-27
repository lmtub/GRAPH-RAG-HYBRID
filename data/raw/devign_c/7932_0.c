static void xilinx_spips_flush_txfifo(XilinxSPIPS *s)

{

    for (;;) {

        int i;

        uint8_t rx;

        uint8_t tx = 0;



        for (i = 0; i < num_effective_busses(s); ++i) {

            if (!i || s->snoop_state == SNOOP_STRIPING) {

                if (fifo8_is_empty(&s->tx_fifo)) {

                    s->regs[R_INTR_STATUS] |= IXR_TX_FIFO_UNDERFLOW;

                    xilinx_spips_update_ixr(s);

                    return;

                } else {

                    tx = fifo8_pop(&s->tx_fifo);

                }

            }

            rx = ssi_transfer(s->spi[i], (uint32_t)tx);

            DB_PRINT("tx = %02x rx = %02x\n", tx, rx);

            if (!i || s->snoop_state == SNOOP_STRIPING) {

                if (fifo8_is_full(&s->rx_fifo)) {

                    s->regs[R_INTR_STATUS] |= IXR_RX_FIFO_OVERFLOW;

                    DB_PRINT("rx FIFO overflow");

                } else {

                    fifo8_push(&s->rx_fifo, (uint8_t)rx);

                }

            }

        }



        switch (s->snoop_state) {

        case (SNOOP_CHECKING):

            switch (tx) { /* new instruction code */

            case 0x0b: /* dual/quad output read DOR/QOR */

            case 0x6b:

                s->snoop_state = 4;

                break;

            /* FIXME: these vary between vendor - set to spansion */

            case 0xbb: /* high performance dual read DIOR */

                s->snoop_state = 4;

                break;

            case 0xeb: /* high performance quad read QIOR */

                s->snoop_state = 6;

                break;

            default:

                s->snoop_state = SNOOP_NONE;

            }

            break;

        case (SNOOP_STRIPING):

        case (SNOOP_NONE):

            break;

        default:

            s->snoop_state--;

        }

    }

}
