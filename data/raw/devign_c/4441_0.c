static void omap1_mpu_reset(void *opaque)

{

    struct omap_mpu_state_s *mpu = (struct omap_mpu_state_s *) opaque;



    omap_inth_reset(mpu->ih[0]);

    omap_inth_reset(mpu->ih[1]);

    omap_dma_reset(mpu->dma);

    omap_mpu_timer_reset(mpu->timer[0]);

    omap_mpu_timer_reset(mpu->timer[1]);

    omap_mpu_timer_reset(mpu->timer[2]);

    omap_wd_timer_reset(mpu->wdt);

    omap_os_timer_reset(mpu->os_timer);

    omap_lcdc_reset(mpu->lcd);

    omap_ulpd_pm_reset(mpu);

    omap_pin_cfg_reset(mpu);

    omap_mpui_reset(mpu);

    omap_tipb_bridge_reset(mpu->private_tipb);

    omap_tipb_bridge_reset(mpu->public_tipb);

    omap_dpll_reset(&mpu->dpll[0]);

    omap_dpll_reset(&mpu->dpll[1]);

    omap_dpll_reset(&mpu->dpll[2]);

    omap_uart_reset(mpu->uart[0]);

    omap_uart_reset(mpu->uart[1]);

    omap_uart_reset(mpu->uart[2]);

    omap_mmc_reset(mpu->mmc);

    omap_mpuio_reset(mpu->mpuio);

    omap_uwire_reset(mpu->microwire);

    omap_pwl_reset(mpu);

    omap_pwt_reset(mpu);

    omap_i2c_reset(mpu->i2c[0]);

    omap_rtc_reset(mpu->rtc);

    omap_mcbsp_reset(mpu->mcbsp1);

    omap_mcbsp_reset(mpu->mcbsp2);

    omap_mcbsp_reset(mpu->mcbsp3);

    omap_lpg_reset(mpu->led[0]);

    omap_lpg_reset(mpu->led[1]);

    omap_clkm_reset(mpu);

    cpu_reset(mpu->env);

}
