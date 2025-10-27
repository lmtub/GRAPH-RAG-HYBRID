static uint64_t omap_lpg_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    struct omap_lpg_s *s = (struct omap_lpg_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 1) {

        return omap_badwidth_read8(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* LCR */

        return s->control;



    case 0x04:	/* PMR */

        return s->power;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
