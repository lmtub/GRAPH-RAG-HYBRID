static uint64_t cs_mem_read(void *opaque, target_phys_addr_t addr,

                            unsigned size)

{

    CSState *s = opaque;

    uint32_t saddr, ret;



    saddr = addr >> 2;

    switch (saddr) {

    case 1:

        switch (CS_RAP(s)) {

        case 3: // Write only

            ret = 0;

            break;

        default:

            ret = s->dregs[CS_RAP(s)];

            break;

        }

        trace_cs4231_mem_readl_dreg(CS_RAP(s), ret);

        break;

    default:

        ret = s->regs[saddr];

        trace_cs4231_mem_readl_reg(saddr, ret);

        break;

    }

    return ret;

}
