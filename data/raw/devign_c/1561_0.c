static void omap2_mpu_reset(void *opaque)

{

    struct omap_mpu_state_s *mpu = (struct omap_mpu_state_s *) opaque;



    omap_inth_reset(mpu->ih[0]);

    omap_dma_reset(mpu->dma);

    omap_prcm_reset(mpu->prcm);

    omap_sysctl_reset(mpu->sysc);

    omap_gp_timer_reset(mpu->gptimer[0]);

    omap_gp_timer_reset(mpu->gptimer[1]);

    omap_gp_timer_reset(mpu->gptimer[2]);

    omap_gp_timer_reset(mpu->gptimer[3]);

    omap_gp_timer_reset(mpu->gptimer[4]);

    omap_gp_timer_reset(mpu->gptimer[5]);

    omap_gp_timer_reset(mpu->gptimer[6]);

    omap_gp_timer_reset(mpu->gptimer[7]);

    omap_gp_timer_reset(mpu->gptimer[8]);

    omap_gp_timer_reset(mpu->gptimer[9]);

    omap_gp_timer_reset(mpu->gptimer[10]);

    omap_gp_timer_reset(mpu->gptimer[11]);

    omap_synctimer_reset(mpu->synctimer);

    omap_sdrc_reset(mpu->sdrc);

    omap_gpmc_reset(mpu->gpmc);

    omap_dss_reset(mpu->dss);

    omap_uart_reset(mpu->uart[0]);

    omap_uart_reset(mpu->uart[1]);

    omap_uart_reset(mpu->uart[2]);

    omap_mmc_reset(mpu->mmc);

    omap_mcspi_reset(mpu->mcspi[0]);

    omap_mcspi_reset(mpu->mcspi[1]);

    omap_i2c_reset(mpu->i2c[0]);

    omap_i2c_reset(mpu->i2c[1]);

    cpu_reset(mpu->env);

}
