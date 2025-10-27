static void lan9118_writel(void *opaque, target_phys_addr_t offset,

                           uint32_t val)

{

    lan9118_state *s = (lan9118_state *)opaque;

    offset &= 0xff;

    

    //DPRINTF("Write reg 0x%02x = 0x%08x\n", (int)offset, val);

    if (offset >= 0x20 && offset < 0x40) {

        /* TX FIFO */

        tx_fifo_push(s, val);

        return;

    }

    switch (offset) {

    case CSR_IRQ_CFG:

        /* TODO: Implement interrupt deassertion intervals.  */

        s->irq_cfg = (s->irq_cfg & IRQ_INT) | (val & IRQ_EN);

        break;

    case CSR_INT_STS:

        s->int_sts &= ~val;

        break;

    case CSR_INT_EN:

        s->int_en = val & ~RESERVED_INT;

        s->int_sts |= val & SW_INT;

        break;

    case CSR_FIFO_INT:

        DPRINTF("FIFO INT levels %08x\n", val);

        s->fifo_int = val;

        break;

    case CSR_RX_CFG:

        if (val & 0x8000) {

            /* RX_DUMP */

            s->rx_fifo_used = 0;

            s->rx_status_fifo_used = 0;

            s->rx_packet_size_tail = s->rx_packet_size_head;

            s->rx_packet_size[s->rx_packet_size_head] = 0;

        }

        s->rx_cfg = val & 0xcfff1ff0;

        break;

    case CSR_TX_CFG:

        if (val & 0x8000) {

            s->tx_status_fifo_used = 0;

        }

        if (val & 0x4000) {

            s->txp->state = TX_IDLE;

            s->txp->fifo_used = 0;

            s->txp->cmd_a = 0xffffffff;

        }

        s->tx_cfg = val & 6;

        break;

    case CSR_HW_CFG:

        if (val & 1) {

            /* SRST */

            lan9118_reset(&s->busdev.qdev);

        } else {

            s->hw_cfg = val & 0x003f300;

        }

        break;

    case CSR_RX_DP_CTRL:

        if (val & 0x80000000) {

            /* Skip forward to next packet.  */

            s->rxp_pad = 0;

            s->rxp_offset = 0;

            if (s->rxp_size == 0) {

                /* Pop a word to start the next packet.  */

                rx_fifo_pop(s);

                s->rxp_pad = 0;

                s->rxp_offset = 0;

            }

            s->rx_fifo_head += s->rxp_size;

            if (s->rx_fifo_head >= s->rx_fifo_size) {

                s->rx_fifo_head -= s->rx_fifo_size;

            }

        }

        break;

    case CSR_PMT_CTRL:

        if (val & 0x400) {

            phy_reset(s);

        }

        s->pmt_ctrl &= ~0x34e;

        s->pmt_ctrl |= (val & 0x34e);

        break;

    case CSR_GPIO_CFG:

        /* Probably just enabling LEDs.  */

        s->gpio_cfg = val & 0x7777071f;

        break;

    case CSR_GPT_CFG:

        if ((s->gpt_cfg ^ val) & GPT_TIMER_EN) {

            if (val & GPT_TIMER_EN) {

                ptimer_set_count(s->timer, val & 0xffff);

                ptimer_run(s->timer, 0);

            } else {

                ptimer_stop(s->timer);

                ptimer_set_count(s->timer, 0xffff);

            }

        }

        s->gpt_cfg = val & (GPT_TIMER_EN | 0xffff);

        break;

    case CSR_WORD_SWAP:

        /* Ignored because we're in 32-bit mode.  */

        s->word_swap = val;

        break;

    case CSR_MAC_CSR_CMD:

        s->mac_cmd = val & 0x4000000f;

        if (val & 0x80000000) {

            if (val & 0x40000000) {

                s->mac_data = do_mac_read(s, val & 0xf);

                DPRINTF("MAC read %d = 0x%08x\n", val & 0xf, s->mac_data);

            } else {

                DPRINTF("MAC write %d = 0x%08x\n", val & 0xf, s->mac_data);

                do_mac_write(s, val & 0xf, s->mac_data);

            }

        }

        break;

    case CSR_MAC_CSR_DATA:

        s->mac_data = val;

        break;

    case CSR_AFC_CFG:

        s->afc_cfg = val & 0x00ffffff;

        break;

    case CSR_E2P_CMD:

        lan9118_eeprom_cmd(s, (val >> 28) & 7, val & 0xff);

        break;

    case CSR_E2P_DATA:

        s->e2p_data = val & 0xff;

        break;



    default:

        hw_error("lan9118_write: Bad reg 0x%x = %x\n", (int)offset, val);

        break;

    }

    lan9118_update(s);

}
