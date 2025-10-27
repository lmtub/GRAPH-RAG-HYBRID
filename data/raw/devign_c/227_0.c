static uint64_t omap_pwt_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    struct omap_pwt_s *s = (struct omap_pwt_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 1) {

        return omap_badwidth_read8(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* FRC */

        return s->frc;

    case 0x04:	/* VCR */

        return s->vrc;

    case 0x08:	/* GCR */

        return s->gcr;

    }

    OMAP_BAD_REG(addr);

    return 0;

}
