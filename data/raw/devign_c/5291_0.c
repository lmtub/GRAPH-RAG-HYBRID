static uint64_t omap_uwire_read(void *opaque, target_phys_addr_t addr,

                                unsigned size)

{

    struct omap_uwire_s *s = (struct omap_uwire_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* RDR */

        s->control &= ~(1 << 15);			/* RDRB */

        return s->rxbuf;



    case 0x04:	/* CSR */

        return s->control;



    case 0x08:	/* SR1 */

        return s->setup[0];

    case 0x0c:	/* SR2 */

        return s->setup[1];

    case 0x10:	/* SR3 */

        return s->setup[2];

    case 0x14:	/* SR4 */

        return s->setup[3];

    case 0x18:	/* SR5 */

        return s->setup[4];

    }



    OMAP_BAD_REG(addr);

    return 0;

}
