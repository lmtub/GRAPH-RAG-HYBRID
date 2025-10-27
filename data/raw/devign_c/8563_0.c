static void omap_prcm_dpll_update(struct omap_prcm_s *s)

{

    omap_clk dpll = omap_findclk(s->mpu, "dpll");

    omap_clk dpll_x2 = omap_findclk(s->mpu, "dpll");

    omap_clk core = omap_findclk(s->mpu, "core_clk");

    int mode = (s->clken[9] >> 0) & 3;

    int mult, div;



    mult = (s->clksel[5] >> 12) & 0x3ff;

    div = (s->clksel[5] >> 8) & 0xf;

    if (mult == 0 || mult == 1)

        mode = 1;	/* Bypass */



    s->dpll_lock = 0;

    switch (mode) {

    case 0:

        fprintf(stderr, "%s: bad EN_DPLL\n", __FUNCTION__);

        break;

    case 1:	/* Low-power bypass mode (Default) */

    case 2:	/* Fast-relock bypass mode */

        omap_clk_setrate(dpll, 1, 1);

        omap_clk_setrate(dpll_x2, 1, 1);

        break;

    case 3:	/* Lock mode */

        s->dpll_lock = 1; /* After 20 FINT cycles (ref_clk / (div + 1)).  */



        omap_clk_setrate(dpll, div + 1, mult);

        omap_clk_setrate(dpll_x2, div + 1, mult * 2);

        break;

    }



    switch ((s->clksel[6] >> 0) & 3) {

    case 0:

        omap_clk_reparent(core, omap_findclk(s->mpu, "clk32-kHz"));

        break;

    case 1:

        omap_clk_reparent(core, dpll);

        break;

    case 2:

        /* Default */

        omap_clk_reparent(core, dpll_x2);

        break;

    case 3:

        fprintf(stderr, "%s: bad CORE_CLK_SRC\n", __FUNCTION__);

        break;

    }

}
