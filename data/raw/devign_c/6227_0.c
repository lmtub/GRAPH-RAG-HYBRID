static void omap_pwt_write(void *opaque, hwaddr addr,

                           uint64_t value, unsigned size)

{

    struct omap_pwt_s *s = (struct omap_pwt_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 1) {

        omap_badwidth_write8(opaque, addr, value);

        return;

    }



    switch (offset) {

    case 0x00:	/* FRC */

        s->frc = value & 0x3f;

        break;

    case 0x04:	/* VRC */

        if ((value ^ s->vrc) & 1) {

            if (value & 1)

                printf("%s: %iHz buzz on\n", __FUNCTION__, (int)

                                /* 1.5 MHz from a 12-MHz or 13-MHz PWT_CLK */

                                ((omap_clk_getrate(s->clk) >> 3) /

                                 /* Pre-multiplexer divider */

                                 ((s->gcr & 2) ? 1 : 154) /

                                 /* Octave multiplexer */

                                 (2 << (value & 3)) *

                                 /* 101/107 divider */

                                 ((value & (1 << 2)) ? 101 : 107) *

                                 /*  49/55 divider */

                                 ((value & (1 << 3)) ?  49 : 55) *

                                 /*  50/63 divider */

                                 ((value & (1 << 4)) ?  50 : 63) *

                                 /*  80/127 divider */

                                 ((value & (1 << 5)) ?  80 : 127) /

                                 (107 * 55 * 63 * 127)));

            else

                printf("%s: silence!\n", __FUNCTION__);

        }

        s->vrc = value & 0x7f;

        break;

    case 0x08:	/* GCR */

        s->gcr = value & 3;

        break;

    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
