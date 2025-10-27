static void tx_fifo_push(lan9118_state *s, uint32_t val)

{

    int n;



    if (s->txp->fifo_used == s->tx_fifo_size) {

        s->int_sts |= TDFO_INT;

        return;

    }

    switch (s->txp->state) {

    case TX_IDLE:

        s->txp->cmd_a = val & 0x831f37ff;

        s->txp->fifo_used++;

        s->txp->state = TX_B;

        s->txp->buffer_size = extract32(s->txp->cmd_a, 0, 11);

        s->txp->offset = extract32(s->txp->cmd_a, 16, 5);

        break;

    case TX_B:

        if (s->txp->cmd_a & 0x2000) {

            /* First segment */

            s->txp->cmd_b = val;

            s->txp->fifo_used++;

            /* End alignment does not include command words.  */

            n = (s->txp->buffer_size + s->txp->offset + 3) >> 2;

            switch ((n >> 24) & 3) {

            case 1:

                n = (-n) & 3;

                break;

            case 2:

                n = (-n) & 7;

                break;

            default:

                n = 0;

            }

            s->txp->pad = n;

            s->txp->len = 0;

        }

        DPRINTF("Block len:%d offset:%d pad:%d cmd %08x\n",

                s->txp->buffer_size, s->txp->offset, s->txp->pad,

                s->txp->cmd_a);

        s->txp->state = TX_DATA;

        break;

    case TX_DATA:

        if (s->txp->offset >= 4) {

            s->txp->offset -= 4;

            break;

        }

        if (s->txp->buffer_size <= 0 && s->txp->pad != 0) {

            s->txp->pad--;

        } else {

            n = 4;

            while (s->txp->offset) {

                val >>= 8;

                n--;

                s->txp->offset--;

            }

            /* Documentation is somewhat unclear on the ordering of bytes

               in FIFO words.  Empirical results show it to be little-endian.

               */

            /* TODO: FIFO overflow checking.  */

            while (n--) {

                s->txp->data[s->txp->len] = val & 0xff;

                s->txp->len++;

                val >>= 8;

                s->txp->buffer_size--;

            }

            s->txp->fifo_used++;

        }

        if (s->txp->buffer_size <= 0 && s->txp->pad == 0) {

            if (s->txp->cmd_a & 0x1000) {

                do_tx_packet(s);

            }

            if (s->txp->cmd_a & 0x80000000) {

                s->int_sts |= TX_IOC_INT;

            }

            s->txp->state = TX_IDLE;

        }

        break;

    }

}
