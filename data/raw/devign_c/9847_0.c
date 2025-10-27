minimac2_write(void *opaque, target_phys_addr_t addr, uint64_t value,

               unsigned size)

{

    MilkymistMinimac2State *s = opaque;



    trace_milkymist_minimac2_memory_read(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_MDIO:

    {

        /* MDIO_DI is read only */

        int mdio_di = (s->regs[R_MDIO] & MDIO_DI);

        s->regs[R_MDIO] = value;

        if (mdio_di) {

            s->regs[R_MDIO] |= mdio_di;

        } else {

            s->regs[R_MDIO] &= ~mdio_di;

        }



        minimac2_update_mdio(s);

    } break;

    case R_TXCOUNT:

        s->regs[addr] = value;

        if (value > 0) {

            minimac2_tx(s);

        }

        break;

    case R_STATE0:

    case R_STATE1:

        s->regs[addr] = value;

        update_rx_interrupt(s);

        break;

    case R_SETUP:

    case R_COUNT0:

    case R_COUNT1:

        s->regs[addr] = value;

        break;



    default:

        error_report("milkymist_minimac2: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}
