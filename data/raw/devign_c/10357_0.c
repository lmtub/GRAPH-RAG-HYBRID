static uint64_t imx_ccm_read(void *opaque, target_phys_addr_t offset,

                                unsigned size)

{

    IMXCCMState *s = (IMXCCMState *)opaque;



    DPRINTF("read(offset=%x)", offset >> 2);

    switch (offset >> 2) {

    case 0: /* CCMR */

        DPRINTF(" ccmr = 0x%x\n", s->ccmr);

        return s->ccmr;

    case 1:

        DPRINTF(" pdr0 = 0x%x\n", s->pdr0);

        return s->pdr0;

    case 2:

        DPRINTF(" pdr1 = 0x%x\n", s->pdr1);

        return s->pdr1;

    case 4:

        DPRINTF(" mpctl = 0x%x\n", s->mpctl);

        return s->mpctl;

    case 6:

        DPRINTF(" spctl = 0x%x\n", s->spctl);

        return s->spctl;

    case 8:

        DPRINTF(" cgr0 = 0x%x\n", s->cgr[0]);

        return s->cgr[0];

    case 9:

        DPRINTF(" cgr1 = 0x%x\n", s->cgr[1]);

        return s->cgr[1];

    case 10:

        DPRINTF(" cgr2 = 0x%x\n", s->cgr[2]);

        return s->cgr[2];

    case 18: /* LTR1 */

        return 0x00004040;

    case 23:

        DPRINTF(" pcmr0 = 0x%x\n", s->pmcr0);

        return s->pmcr0;

    }

    DPRINTF(" return 0\n");

    return 0;

}
