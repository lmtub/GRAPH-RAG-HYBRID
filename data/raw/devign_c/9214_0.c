static void scoop_writeb(void *opaque, target_phys_addr_t addr, uint32_t value)

{

    ScoopInfo *s = (ScoopInfo *) opaque;

    value &= 0xffff;



    switch (addr) {

    case SCOOP_MCR:

        s->mcr = value;

        break;

    case SCOOP_CDR:

        s->cdr = value;

        break;

    case SCOOP_CPR:

        s->power = value;

        if (value & 0x80)

            s->power |= 0x8040;

        break;

    case SCOOP_CCR:

        s->ccr = value;

        break;

    case SCOOP_IRR_IRM:

        s->irr = value;

        break;

    case SCOOP_IMR:

        s->imr = value;

        break;

    case SCOOP_ISR:

        s->isr = value;

        break;

    case SCOOP_GPCR:

        s->gpio_dir = value;

        scoop_gpio_handler_update(s);

        break;

    case SCOOP_GPWR:

    case SCOOP_GPRR:	/* GPRR is probably R/O in real HW */

        s->gpio_level = value & s->gpio_dir;

        scoop_gpio_handler_update(s);

        break;

    default:

        zaurus_printf("Bad register offset " REG_FMT "\n", (unsigned long)addr);

    }

}
