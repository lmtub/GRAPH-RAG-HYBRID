static uint64_t omap_prcm_read(void *opaque, target_phys_addr_t addr,

                               unsigned size)

{

    struct omap_prcm_s *s = (struct omap_prcm_s *) opaque;

    uint32_t ret;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0x000:	/* PRCM_REVISION */

        return 0x10;



    case 0x010:	/* PRCM_SYSCONFIG */

        return s->sysconfig;



    case 0x018:	/* PRCM_IRQSTATUS_MPU */

        return s->irqst[0];



    case 0x01c:	/* PRCM_IRQENABLE_MPU */

        return s->irqen[0];



    case 0x050:	/* PRCM_VOLTCTRL */

        return s->voltctrl;

    case 0x054:	/* PRCM_VOLTST */

        return s->voltctrl & 3;



    case 0x060:	/* PRCM_CLKSRC_CTRL */

        return s->clksrc[0];

    case 0x070:	/* PRCM_CLKOUT_CTRL */

        return s->clkout[0];

    case 0x078:	/* PRCM_CLKEMUL_CTRL */

        return s->clkemul[0];

    case 0x080:	/* PRCM_CLKCFG_CTRL */

    case 0x084:	/* PRCM_CLKCFG_STATUS */

        return 0;



    case 0x090:	/* PRCM_VOLTSETUP */

        return s->setuptime[0];



    case 0x094:	/* PRCM_CLKSSETUP */

        return s->setuptime[1];



    case 0x098:	/* PRCM_POLCTRL */

        return s->clkpol[0];



    case 0x0b0:	/* GENERAL_PURPOSE1 */

    case 0x0b4:	/* GENERAL_PURPOSE2 */

    case 0x0b8:	/* GENERAL_PURPOSE3 */

    case 0x0bc:	/* GENERAL_PURPOSE4 */

    case 0x0c0:	/* GENERAL_PURPOSE5 */

    case 0x0c4:	/* GENERAL_PURPOSE6 */

    case 0x0c8:	/* GENERAL_PURPOSE7 */

    case 0x0cc:	/* GENERAL_PURPOSE8 */

    case 0x0d0:	/* GENERAL_PURPOSE9 */

    case 0x0d4:	/* GENERAL_PURPOSE10 */

    case 0x0d8:	/* GENERAL_PURPOSE11 */

    case 0x0dc:	/* GENERAL_PURPOSE12 */

    case 0x0e0:	/* GENERAL_PURPOSE13 */

    case 0x0e4:	/* GENERAL_PURPOSE14 */

    case 0x0e8:	/* GENERAL_PURPOSE15 */

    case 0x0ec:	/* GENERAL_PURPOSE16 */

    case 0x0f0:	/* GENERAL_PURPOSE17 */

    case 0x0f4:	/* GENERAL_PURPOSE18 */

    case 0x0f8:	/* GENERAL_PURPOSE19 */

    case 0x0fc:	/* GENERAL_PURPOSE20 */

        return s->scratch[(addr - 0xb0) >> 2];



    case 0x140:	/* CM_CLKSEL_MPU */

        return s->clksel[0];

    case 0x148:	/* CM_CLKSTCTRL_MPU */

        return s->clkctrl[0];



    case 0x158:	/* RM_RSTST_MPU */

        return s->rst[0];

    case 0x1c8:	/* PM_WKDEP_MPU */

        return s->wkup[0];

    case 0x1d4:	/* PM_EVGENCTRL_MPU */

        return s->ev;

    case 0x1d8:	/* PM_EVEGENONTIM_MPU */

        return s->evtime[0];

    case 0x1dc:	/* PM_EVEGENOFFTIM_MPU */

        return s->evtime[1];

    case 0x1e0:	/* PM_PWSTCTRL_MPU */

        return s->power[0];

    case 0x1e4:	/* PM_PWSTST_MPU */

        return 0;



    case 0x200:	/* CM_FCLKEN1_CORE */

        return s->clken[0];

    case 0x204:	/* CM_FCLKEN2_CORE */

        return s->clken[1];

    case 0x210:	/* CM_ICLKEN1_CORE */

        return s->clken[2];

    case 0x214:	/* CM_ICLKEN2_CORE */

        return s->clken[3];

    case 0x21c:	/* CM_ICLKEN4_CORE */

        return s->clken[4];



    case 0x220:	/* CM_IDLEST1_CORE */

        /* TODO: check the actual iclk status */

        return 0x7ffffff9;

    case 0x224:	/* CM_IDLEST2_CORE */

        /* TODO: check the actual iclk status */

        return 0x00000007;

    case 0x22c:	/* CM_IDLEST4_CORE */

        /* TODO: check the actual iclk status */

        return 0x0000001f;



    case 0x230:	/* CM_AUTOIDLE1_CORE */

        return s->clkidle[0];

    case 0x234:	/* CM_AUTOIDLE2_CORE */

        return s->clkidle[1];

    case 0x238:	/* CM_AUTOIDLE3_CORE */

        return s->clkidle[2];

    case 0x23c:	/* CM_AUTOIDLE4_CORE */

        return s->clkidle[3];



    case 0x240:	/* CM_CLKSEL1_CORE */

        return s->clksel[1];

    case 0x244:	/* CM_CLKSEL2_CORE */

        return s->clksel[2];



    case 0x248:	/* CM_CLKSTCTRL_CORE */

        return s->clkctrl[1];



    case 0x2a0:	/* PM_WKEN1_CORE */

        return s->wken[0];

    case 0x2a4:	/* PM_WKEN2_CORE */

        return s->wken[1];



    case 0x2b0:	/* PM_WKST1_CORE */

        return s->wkst[0];

    case 0x2b4:	/* PM_WKST2_CORE */

        return s->wkst[1];

    case 0x2c8:	/* PM_WKDEP_CORE */

        return 0x1e;



    case 0x2e0:	/* PM_PWSTCTRL_CORE */

        return s->power[1];

    case 0x2e4:	/* PM_PWSTST_CORE */

        return 0x000030 | (s->power[1] & 0xfc00);



    case 0x300:	/* CM_FCLKEN_GFX */

        return s->clken[5];

    case 0x310:	/* CM_ICLKEN_GFX */

        return s->clken[6];

    case 0x320:	/* CM_IDLEST_GFX */

        /* TODO: check the actual iclk status */

        return 0x00000001;

    case 0x340:	/* CM_CLKSEL_GFX */

        return s->clksel[3];

    case 0x348:	/* CM_CLKSTCTRL_GFX */

        return s->clkctrl[2];

    case 0x350:	/* RM_RSTCTRL_GFX */

        return s->rstctrl[0];

    case 0x358:	/* RM_RSTST_GFX */

        return s->rst[1];

    case 0x3c8:	/* PM_WKDEP_GFX */

        return s->wkup[1];



    case 0x3e0:	/* PM_PWSTCTRL_GFX */

        return s->power[2];

    case 0x3e4:	/* PM_PWSTST_GFX */

        return s->power[2] & 3;



    case 0x400:	/* CM_FCLKEN_WKUP */

        return s->clken[7];

    case 0x410:	/* CM_ICLKEN_WKUP */

        return s->clken[8];

    case 0x420:	/* CM_IDLEST_WKUP */

        /* TODO: check the actual iclk status */

        return 0x0000003f;

    case 0x430:	/* CM_AUTOIDLE_WKUP */

        return s->clkidle[4];

    case 0x440:	/* CM_CLKSEL_WKUP */

        return s->clksel[4];

    case 0x450:	/* RM_RSTCTRL_WKUP */

        return 0;

    case 0x454:	/* RM_RSTTIME_WKUP */

        return s->rsttime_wkup;

    case 0x458:	/* RM_RSTST_WKUP */

        return s->rst[2];

    case 0x4a0:	/* PM_WKEN_WKUP */

        return s->wken[2];

    case 0x4b0:	/* PM_WKST_WKUP */

        return s->wkst[2];



    case 0x500:	/* CM_CLKEN_PLL */

        return s->clken[9];

    case 0x520:	/* CM_IDLEST_CKGEN */

        ret = 0x0000070 | (s->apll_lock[0] << 9) | (s->apll_lock[1] << 8);

        if (!(s->clksel[6] & 3))

            /* Core uses 32-kHz clock */

            ret |= 3 << 0;

        else if (!s->dpll_lock)

            /* DPLL not locked, core uses ref_clk */

            ret |= 1 << 0;

        else

            /* Core uses DPLL */

            ret |= 2 << 0;

        return ret;

    case 0x530:	/* CM_AUTOIDLE_PLL */

        return s->clkidle[5];

    case 0x540:	/* CM_CLKSEL1_PLL */

        return s->clksel[5];

    case 0x544:	/* CM_CLKSEL2_PLL */

        return s->clksel[6];



    case 0x800:	/* CM_FCLKEN_DSP */

        return s->clken[10];

    case 0x810:	/* CM_ICLKEN_DSP */

        return s->clken[11];

    case 0x820:	/* CM_IDLEST_DSP */

        /* TODO: check the actual iclk status */

        return 0x00000103;

    case 0x830:	/* CM_AUTOIDLE_DSP */

        return s->clkidle[6];

    case 0x840:	/* CM_CLKSEL_DSP */

        return s->clksel[7];

    case 0x848:	/* CM_CLKSTCTRL_DSP */

        return s->clkctrl[3];

    case 0x850:	/* RM_RSTCTRL_DSP */

        return 0;

    case 0x858:	/* RM_RSTST_DSP */

        return s->rst[3];

    case 0x8c8:	/* PM_WKDEP_DSP */

        return s->wkup[2];

    case 0x8e0:	/* PM_PWSTCTRL_DSP */

        return s->power[3];

    case 0x8e4:	/* PM_PWSTST_DSP */

        return 0x008030 | (s->power[3] & 0x3003);



    case 0x8f0:	/* PRCM_IRQSTATUS_DSP */

        return s->irqst[1];

    case 0x8f4:	/* PRCM_IRQENABLE_DSP */

        return s->irqen[1];



    case 0x8f8:	/* PRCM_IRQSTATUS_IVA */

        return s->irqst[2];

    case 0x8fc:	/* PRCM_IRQENABLE_IVA */

        return s->irqen[2];

    }



    OMAP_BAD_REG(addr);

    return 0;

}
