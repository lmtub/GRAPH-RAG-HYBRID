static void omap_dpll_write(void *opaque, target_phys_addr_t addr,

                            uint64_t value, unsigned size)

{

    struct dpll_ctl_s *s = (struct dpll_ctl_s *) opaque;

    uint16_t diff;

    static const int bypass_div[4] = { 1, 2, 4, 4 };

    int div, mult;



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    if (addr == 0x00) {	/* CTL_REG */

        /* See omap_ulpd_pm_write() too */

        diff = s->mode & value;

        s->mode = value & 0x2fff;

        if (diff & (0x3ff << 2)) {

            if (value & (1 << 4)) {			/* PLL_ENABLE */

                div = ((value >> 5) & 3) + 1;		/* PLL_DIV */

                mult = MIN((value >> 7) & 0x1f, 1);	/* PLL_MULT */

            } else {

                div = bypass_div[((value >> 2) & 3)];	/* BYPASS_DIV */

                mult = 1;

            }

            omap_clk_setrate(s->dpll, div, mult);

        }



        /* Enter the desired mode.  */

        s->mode = (s->mode & 0xfffe) | ((s->mode >> 4) & 1);



        /* Act as if the lock is restored.  */

        s->mode |= 2;

    } else {

        OMAP_BAD_REG(addr);

    }

}
