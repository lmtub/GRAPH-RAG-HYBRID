minimac2_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    MilkymistMinimac2State *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_SETUP:

    case R_MDIO:

    case R_STATE0:

    case R_COUNT0:

    case R_STATE1:

    case R_COUNT1:

    case R_TXCOUNT:

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_minimac2: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_minimac2_memory_read(addr << 2, r);



    return r;

}
