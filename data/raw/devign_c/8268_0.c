static void gem_write(void *opaque, hwaddr offset, uint64_t val,

        unsigned size)

{

    GemState *s = (GemState *)opaque;

    uint32_t readonly;



    DB_PRINT("offset: 0x%04x write: 0x%08x ", (unsigned)offset, (unsigned)val);

    offset >>= 2;



    /* Squash bits which are read only in write value */

    val &= ~(s->regs_ro[offset]);

    /* Preserve (only) bits which are read only and wtc in register */

    readonly = s->regs[offset] & (s->regs_ro[offset] | s->regs_w1c[offset]);



    /* Copy register write to backing store */

    s->regs[offset] = (val & ~s->regs_w1c[offset]) | readonly;



    /* do w1c */

    s->regs[offset] &= ~(s->regs_w1c[offset] & val);



    /* Handle register write side effects */

    switch (offset) {

    case GEM_NWCTRL:

        if (val & GEM_NWCTRL_RXENA) {

            gem_get_rx_desc(s);

        }

        if (val & GEM_NWCTRL_TXSTART) {

            gem_transmit(s);

        }

        if (!(val & GEM_NWCTRL_TXENA)) {

            /* Reset to start of Q when transmit disabled. */

            s->tx_desc_addr = s->regs[GEM_TXQBASE];

        }

        if (val & GEM_NWCTRL_RXENA) {

            qemu_flush_queued_packets(qemu_get_queue(s->nic));

        }

        break;



    case GEM_TXSTATUS:

        gem_update_int_status(s);

        break;

    case GEM_RXQBASE:

        s->rx_desc_addr = val;

        break;

    case GEM_TXQBASE:

        s->tx_desc_addr = val;

        break;

    case GEM_RXSTATUS:

        gem_update_int_status(s);

        break;

    case GEM_IER:

        s->regs[GEM_IMR] &= ~val;

        gem_update_int_status(s);

        break;

    case GEM_IDR:

        s->regs[GEM_IMR] |= val;

        gem_update_int_status(s);

        break;

    case GEM_SPADDR1LO:

    case GEM_SPADDR2LO:

    case GEM_SPADDR3LO:

    case GEM_SPADDR4LO:

        s->sar_active[(offset - GEM_SPADDR1LO) / 2] = false;

        break;

    case GEM_SPADDR1HI:

    case GEM_SPADDR2HI:

    case GEM_SPADDR3HI:

    case GEM_SPADDR4HI:

        s->sar_active[(offset - GEM_SPADDR1HI) / 2] = true;

        break;

    case GEM_PHYMNTNC:

        if (val & GEM_PHYMNTNC_OP_W) {

            uint32_t phy_addr, reg_num;



            phy_addr = (val & GEM_PHYMNTNC_ADDR) >> GEM_PHYMNTNC_ADDR_SHFT;

            if (phy_addr == BOARD_PHY_ADDRESS) {

                reg_num = (val & GEM_PHYMNTNC_REG) >> GEM_PHYMNTNC_REG_SHIFT;

                gem_phy_write(s, reg_num, val);

            }

        }

        break;

    }



    DB_PRINT("newval: 0x%08x\n", s->regs[offset]);

}
