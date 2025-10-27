static uint32_t scoop_readb(void *opaque, target_phys_addr_t addr)

{

    ScoopInfo *s = (ScoopInfo *) opaque;



    switch (addr) {

    case SCOOP_MCR:

        return s->mcr;

    case SCOOP_CDR:

        return s->cdr;

    case SCOOP_CSR:

        return s->status;

    case SCOOP_CPR:

        return s->power;

    case SCOOP_CCR:

        return s->ccr;

    case SCOOP_IRR_IRM:

        return s->irr;

    case SCOOP_IMR:

        return s->imr;

    case SCOOP_ISR:

        return s->isr;

    case SCOOP_GPCR:

        return s->gpio_dir;

    case SCOOP_GPWR:

    case SCOOP_GPRR:

        return s->gpio_level;

    default:

        zaurus_printf("Bad register offset " REG_FMT "\n", (unsigned long)addr);

    }



    return 0;

}
