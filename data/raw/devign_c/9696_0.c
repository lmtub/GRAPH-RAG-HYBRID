static void omap_os_timer_write(void *opaque, target_phys_addr_t addr,

                                uint64_t value, unsigned size)

{

    struct omap_32khz_timer_s *s = (struct omap_32khz_timer_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 4) {

        return omap_badwidth_write32(opaque, addr, value);

    }



    switch (offset) {

    case 0x00:	/* TVR */

        s->timer.reset_val = value & 0x00ffffff;

        break;



    case 0x04:	/* TCR */

        OMAP_RO_REG(addr);

        break;



    case 0x08:	/* CR */

        s->timer.ar = (value >> 3) & 1;

        s->timer.it_ena = (value >> 2) & 1;

        if (s->timer.st != (value & 1) || (value & 2)) {

            omap_timer_sync(&s->timer);

            s->timer.enable = value & 1;

            s->timer.st = value & 1;

            omap_timer_update(&s->timer);

        }

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
