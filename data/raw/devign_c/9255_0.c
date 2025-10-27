static uint64_t lan9118_readl(void *opaque, target_phys_addr_t offset,

                              unsigned size)

{

    lan9118_state *s = (lan9118_state *)opaque;



    //DPRINTF("Read reg 0x%02x\n", (int)offset);

    if (offset < 0x20) {

        /* RX FIFO */

        return rx_fifo_pop(s);

    }

    switch (offset) {

    case 0x40:

        return rx_status_fifo_pop(s);

    case 0x44:

        return s->rx_status_fifo[s->tx_status_fifo_head];

    case 0x48:

        return tx_status_fifo_pop(s);

    case 0x4c:

        return s->tx_status_fifo[s->tx_status_fifo_head];

    case CSR_ID_REV:

        return 0x01180001;

    case CSR_IRQ_CFG:

        return s->irq_cfg;

    case CSR_INT_STS:

        return s->int_sts;

    case CSR_INT_EN:

        return s->int_en;

    case CSR_BYTE_TEST:

        return 0x87654321;

    case CSR_FIFO_INT:

        return s->fifo_int;

    case CSR_RX_CFG:

        return s->rx_cfg;

    case CSR_TX_CFG:

        return s->tx_cfg;

    case CSR_HW_CFG:

        return s->hw_cfg;

    case CSR_RX_DP_CTRL:

        return 0;

    case CSR_RX_FIFO_INF:

        return (s->rx_status_fifo_used << 16) | (s->rx_fifo_used << 2);

    case CSR_TX_FIFO_INF:

        return (s->tx_status_fifo_used << 16)

               | (s->tx_fifo_size - s->txp->fifo_used);

    case CSR_PMT_CTRL:

        return s->pmt_ctrl;

    case CSR_GPIO_CFG:

        return s->gpio_cfg;

    case CSR_GPT_CFG:

        return s->gpt_cfg;

    case CSR_GPT_CNT:

        return ptimer_get_count(s->timer);

    case CSR_WORD_SWAP:

        return s->word_swap;

    case CSR_FREE_RUN:

        return (qemu_get_clock_ns(vm_clock) / 40) - s->free_timer_start;

    case CSR_RX_DROP:

        /* TODO: Implement dropped frames counter.  */

        return 0;

    case CSR_MAC_CSR_CMD:

        return s->mac_cmd;

    case CSR_MAC_CSR_DATA:

        return s->mac_data;

    case CSR_AFC_CFG:

        return s->afc_cfg;

    case CSR_E2P_CMD:

        return s->e2p_cmd;

    case CSR_E2P_DATA:

        return s->e2p_data;

    }

    hw_error("lan9118_read: Bad reg 0x%x\n", (int)offset);

    return 0;

}
