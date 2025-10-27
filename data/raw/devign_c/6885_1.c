static void stellaris_enet_write(void *opaque, hwaddr offset,

                                 uint64_t value, unsigned size)

{

    stellaris_enet_state *s = (stellaris_enet_state *)opaque;



    switch (offset) {

    case 0x00: /* IACK */

        s->ris &= ~value;

        DPRINTF("IRQ ack %02x/%02x\n", value, s->ris);

        stellaris_enet_update(s);

        /* Clearing TXER also resets the TX fifo.  */

        if (value & SE_INT_TXER)

            s->tx_frame_len = -1;

        break;

    case 0x04: /* IM */

        DPRINTF("IRQ mask %02x/%02x\n", value, s->ris);

        s->im = value;

        stellaris_enet_update(s);

        break;

    case 0x08: /* RCTL */

        s->rctl = value;

        if (value & SE_RCTL_RSTFIFO) {

            s->rx_fifo_len = 0;

            s->np = 0;

            stellaris_enet_update(s);

        }

        break;

    case 0x0c: /* TCTL */

        s->tctl = value;

        break;

    case 0x10: /* DATA */

        if (s->tx_frame_len == -1) {

            s->tx_frame_len = value & 0xffff;

            if (s->tx_frame_len > 2032) {

                DPRINTF("TX frame too long (%d)\n", s->tx_frame_len);

                s->tx_frame_len = 0;

                s->ris |= SE_INT_TXER;

                stellaris_enet_update(s);

            } else {

                DPRINTF("Start TX frame len=%d\n", s->tx_frame_len);

                /* The value written does not include the ethernet header.  */

                s->tx_frame_len += 14;

                if ((s->tctl & SE_TCTL_CRC) == 0)

                    s->tx_frame_len += 4;

                s->tx_fifo_len = 0;

                s->tx_fifo[s->tx_fifo_len++] = value >> 16;

                s->tx_fifo[s->tx_fifo_len++] = value >> 24;

            }

        } else {

            s->tx_fifo[s->tx_fifo_len++] = value;

            s->tx_fifo[s->tx_fifo_len++] = value >> 8;

            s->tx_fifo[s->tx_fifo_len++] = value >> 16;

            s->tx_fifo[s->tx_fifo_len++] = value >> 24;

            if (s->tx_fifo_len >= s->tx_frame_len) {

                /* We don't implement explicit CRC, so just chop it off.  */

                if ((s->tctl & SE_TCTL_CRC) == 0)

                    s->tx_frame_len -= 4;

                if ((s->tctl & SE_TCTL_PADEN) && s->tx_frame_len < 60) {

                    memset(&s->tx_fifo[s->tx_frame_len], 0, 60 - s->tx_frame_len);

                    s->tx_fifo_len = 60;

                }

                qemu_send_packet(qemu_get_queue(s->nic), s->tx_fifo,

                                 s->tx_frame_len);

                s->tx_frame_len = -1;

                s->ris |= SE_INT_TXEMP;

                stellaris_enet_update(s);

                DPRINTF("Done TX\n");

            }

        }

        break;

    case 0x14: /* IA0 */

        s->conf.macaddr.a[0] = value;

        s->conf.macaddr.a[1] = value >> 8;

        s->conf.macaddr.a[2] = value >> 16;

        s->conf.macaddr.a[3] = value >> 24;

        break;

    case 0x18: /* IA1 */

        s->conf.macaddr.a[4] = value;

        s->conf.macaddr.a[5] = value >> 8;

        break;

    case 0x1c: /* THR */

        s->thr = value;

        break;

    case 0x20: /* MCTL */

        s->mctl = value;

        break;

    case 0x24: /* MDV */

        s->mdv = value;

        break;

    case 0x28: /* MADD */

        /* ignored.  */

        break;

    case 0x2c: /* MTXD */

        s->mtxd = value & 0xff;

        break;

    case 0x30: /* MRXD */

    case 0x34: /* NP */

    case 0x38: /* TR */

        /* Ignored.  */

    case 0x3c: /* Undocuented: Timestamp? */

        /* Ignored.  */

        break;

    default:

        hw_error("stellaris_enet_write: Bad offset %x\n", (int)offset);

    }

}
