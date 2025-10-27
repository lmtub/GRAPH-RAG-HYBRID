static void xilinx_spips_flush_txfifo(XilinxSPIPS *s)

{

    int debug_level = 0;

    XilinxQSPIPS *q = (XilinxQSPIPS *) object_dynamic_cast(OBJECT(s),

                                                           TYPE_XILINX_QSPIPS);



    for (;;) {

        int i;

        uint8_t tx = 0;

        uint8_t tx_rx[num_effective_busses(s)];

        uint8_t dummy_cycles = 0;

        uint8_t addr_length;



        if (fifo8_is_empty(&s->tx_fifo)) {

            if (!(s->regs[R_LQSPI_CFG] & LQSPI_CFG_LQ_MODE)) {

                s->regs[R_INTR_STATUS] |= IXR_TX_FIFO_UNDERFLOW;

            }

            xilinx_spips_update_ixr(s);

            return;

        } else if (s->snoop_state == SNOOP_STRIPING) {

            for (i = 0; i < num_effective_busses(s); ++i) {

                tx_rx[i] = fifo8_pop(&s->tx_fifo);

            }

            stripe8(tx_rx, num_effective_busses(s), false);

        } else if (s->snoop_state >= SNOOP_ADDR) {

            tx = fifo8_pop(&s->tx_fifo);

            for (i = 0; i < num_effective_busses(s); ++i) {

                tx_rx[i] = tx;

            }

        } else {

            /* Extract a dummy byte and generate dummy cycles according to the

             * link state */

            tx = fifo8_pop(&s->tx_fifo);

            dummy_cycles = 8 / s->link_state;

        }



        for (i = 0; i < num_effective_busses(s); ++i) {

            int bus = num_effective_busses(s) - 1 - i;

            if (dummy_cycles) {

                int d;

                for (d = 0; d < dummy_cycles; ++d) {

                    tx_rx[0] = ssi_transfer(s->spi[bus], (uint32_t)tx_rx[0]);

                }

            } else {

                DB_PRINT_L(debug_level, "tx = %02x\n", tx_rx[i]);

                tx_rx[i] = ssi_transfer(s->spi[bus], (uint32_t)tx_rx[i]);

                DB_PRINT_L(debug_level, "rx = %02x\n", tx_rx[i]);

            }

        }



        if (s->regs[R_CMND] & R_CMND_RXFIFO_DRAIN) {

            DB_PRINT_L(debug_level, "dircarding drained rx byte\n");

            /* Do nothing */

        } else if (s->rx_discard) {

            DB_PRINT_L(debug_level, "dircarding discarded rx byte\n");

            s->rx_discard -= 8 / s->link_state;

        } else if (fifo8_is_full(&s->rx_fifo)) {

            s->regs[R_INTR_STATUS] |= IXR_RX_FIFO_OVERFLOW;

            DB_PRINT_L(0, "rx FIFO overflow");

        } else if (s->snoop_state == SNOOP_STRIPING) {

            stripe8(tx_rx, num_effective_busses(s), true);

            for (i = 0; i < num_effective_busses(s); ++i) {

                fifo8_push(&s->rx_fifo, (uint8_t)tx_rx[i]);

                DB_PRINT_L(debug_level, "pushing striped rx byte\n");

            }

        } else {

           DB_PRINT_L(debug_level, "pushing unstriped rx byte\n");

           fifo8_push(&s->rx_fifo, (uint8_t)tx_rx[0]);

        }



        if (s->link_state_next_when) {

            s->link_state_next_when--;

            if (!s->link_state_next_when) {

                s->link_state = s->link_state_next;

            }

        }



        DB_PRINT_L(debug_level, "initial snoop state: %x\n",

                   (unsigned)s->snoop_state);

        switch (s->snoop_state) {

        case (SNOOP_CHECKING):

            /* Store the count of dummy bytes in the txfifo */

            s->cmd_dummies = xilinx_spips_num_dummies(q, tx);

            addr_length = get_addr_length(s, tx);

            if (s->cmd_dummies < 0) {

                s->snoop_state = SNOOP_NONE;

            } else {

                s->snoop_state = SNOOP_ADDR + addr_length - 1;

            }

            switch (tx) {

            case DPP:

            case DOR:

            case DOR_4:

                s->link_state_next = 2;

                s->link_state_next_when = addr_length + s->cmd_dummies;

                break;

            case QPP:

            case QPP_4:

            case QOR:

            case QOR_4:

                s->link_state_next = 4;

                s->link_state_next_when = addr_length + s->cmd_dummies;

                break;

            case DIOR:

            case DIOR_4:

                s->link_state = 2;

                break;

            case QIOR:

            case QIOR_4:

                s->link_state = 4;

                break;

            }

            break;

        case (SNOOP_ADDR):

            /* Address has been transmitted, transmit dummy cycles now if

             * needed */

            if (s->cmd_dummies < 0) {

                s->snoop_state = SNOOP_NONE;

            } else {

                s->snoop_state = s->cmd_dummies;

            }

            break;

        case (SNOOP_STRIPING):

        case (SNOOP_NONE):

            /* Once we hit the boring stuff - squelch debug noise */

            if (!debug_level) {

                DB_PRINT_L(0, "squelching debug info ....\n");

                debug_level = 1;

            }

            break;

        default:

            s->snoop_state--;

        }

        DB_PRINT_L(debug_level, "final snoop state: %x\n",

                   (unsigned)s->snoop_state);

    }

}
