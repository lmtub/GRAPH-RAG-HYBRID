static uint64_t pxa2xx_mm_read(void *opaque, hwaddr addr,

                               unsigned size)

{

    PXA2xxState *s = (PXA2xxState *) opaque;



    switch (addr) {

    case MDCNFG ... SA1110:

        if ((addr & 3) == 0)

            return s->mm_regs[addr >> 2];



    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
