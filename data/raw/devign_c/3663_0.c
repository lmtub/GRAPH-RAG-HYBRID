static uint64_t pxa2xx_pm_read(void *opaque, hwaddr addr,

                               unsigned size)

{

    PXA2xxState *s = (PXA2xxState *) opaque;



    switch (addr) {

    case PMCR ... PCMD31:

        if (addr & 3)

            goto fail;



        return s->pm_regs[addr >> 2];

    default:

    fail:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
