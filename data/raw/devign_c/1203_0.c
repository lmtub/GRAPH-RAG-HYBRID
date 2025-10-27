static uint64_t omap_os_timer_read(void *opaque, target_phys_addr_t addr,

                                   unsigned size)

{

    struct omap_32khz_timer_s *s = (struct omap_32khz_timer_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* TVR */

        return s->timer.reset_val;



    case 0x04:	/* TCR */

        return omap_timer_read(&s->timer);



    case 0x08:	/* CR */

        return (s->timer.ar << 3) | (s->timer.it_ena << 2) | s->timer.st;



    default:

        break;

    }

    OMAP_BAD_REG(addr);

    return 0;

}
