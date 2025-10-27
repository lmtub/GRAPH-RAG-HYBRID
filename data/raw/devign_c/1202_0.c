static void dbdma_writel (void *opaque,

                          target_phys_addr_t addr, uint32_t value)

{

    int channel = addr >> DBDMA_CHANNEL_SHIFT;

    DBDMA_channel *ch = (DBDMA_channel *)opaque + channel;

    int reg = (addr - (channel << DBDMA_CHANNEL_SHIFT)) >> 2;



    DBDMA_DPRINTF("writel 0x" TARGET_FMT_plx " <= 0x%08x\n", addr, value);

    DBDMA_DPRINTF("channel 0x%x reg 0x%x\n",

                  (uint32_t)addr >> DBDMA_CHANNEL_SHIFT, reg);



    /* cmdptr cannot be modified if channel is RUN or ACTIVE */



    if (reg == DBDMA_CMDPTR_LO &&

        (ch->regs[DBDMA_STATUS] & cpu_to_be32(RUN | ACTIVE)))

	return;



    ch->regs[reg] = value;



    switch(reg) {

    case DBDMA_CONTROL:

        dbdma_control_write(ch);

        break;

    case DBDMA_CMDPTR_LO:

        /* 16-byte aligned */

        ch->regs[DBDMA_CMDPTR_LO] &= cpu_to_be32(~0xf);

        dbdma_cmdptr_load(ch);

        break;

    case DBDMA_STATUS:

    case DBDMA_INTR_SEL:

    case DBDMA_BRANCH_SEL:

    case DBDMA_WAIT_SEL:

        /* nothing to do */

        break;

    case DBDMA_XFER_MODE:

    case DBDMA_CMDPTR_HI:

    case DBDMA_DATA2PTR_HI:

    case DBDMA_DATA2PTR_LO:

    case DBDMA_ADDRESS_HI:

    case DBDMA_BRANCH_ADDR_HI:

    case DBDMA_RES1:

    case DBDMA_RES2:

    case DBDMA_RES3:

    case DBDMA_RES4:

        /* unused */

        break;

    }

}
