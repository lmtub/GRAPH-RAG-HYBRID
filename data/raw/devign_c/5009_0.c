struct omap_mpu_state_s *omap2420_mpu_init(MemoryRegion *sysmem,

                unsigned long sdram_size,

                const char *core)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *)

            g_malloc0(sizeof(struct omap_mpu_state_s));

    qemu_irq dma_irqs[4];

    DriveInfo *dinfo;

    int i;

    SysBusDevice *busdev;

    struct omap_target_agent_s *ta;



    /* Core */

    s->mpu_model = omap2420;

    s->cpu = cpu_arm_init(core ?: "arm1136-r2");

    if (s->cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    s->sdram_size = sdram_size;

    s->sram_size = OMAP242X_SRAM_SIZE;



    s->wakeup = qemu_allocate_irq(omap_mpu_wakeup, s, 0);



    /* Clocks */

    omap_clk_init(s);



    /* Memory-mapped stuff */

    memory_region_init_ram(&s->sdram, NULL, "omap2.dram", s->sdram_size,

                           &error_abort);

    vmstate_register_ram_global(&s->sdram);

    memory_region_add_subregion(sysmem, OMAP2_Q2_BASE, &s->sdram);

    memory_region_init_ram(&s->sram, NULL, "omap2.sram", s->sram_size,

                           &error_abort);

    vmstate_register_ram_global(&s->sram);

    memory_region_add_subregion(sysmem, OMAP2_SRAM_BASE, &s->sram);



    s->l4 = omap_l4_init(sysmem, OMAP2_L4_BASE, 54);



    /* Actually mapped at any 2K boundary in the ARM11 private-peripheral if */

    s->ih[0] = qdev_create(NULL, "omap2-intc");

    qdev_prop_set_uint8(s->ih[0], "revision", 0x21);

    qdev_prop_set_ptr(s->ih[0], "fclk", omap_findclk(s, "mpu_intc_fclk"));

    qdev_prop_set_ptr(s->ih[0], "iclk", omap_findclk(s, "mpu_intc_iclk"));

    qdev_init_nofail(s->ih[0]);

    busdev = SYS_BUS_DEVICE(s->ih[0]);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(DEVICE(s->cpu), ARM_CPU_IRQ));

    sysbus_connect_irq(busdev, 1,

                       qdev_get_gpio_in(DEVICE(s->cpu), ARM_CPU_FIQ));

    sysbus_mmio_map(busdev, 0, 0x480fe000);

    s->prcm = omap_prcm_init(omap_l4tao(s->l4, 3),

                             qdev_get_gpio_in(s->ih[0],

                                              OMAP_INT_24XX_PRCM_MPU_IRQ),

                             NULL, NULL, s);



    s->sysc = omap_sysctl_init(omap_l4tao(s->l4, 1),

                    omap_findclk(s, "omapctrl_iclk"), s);



    for (i = 0; i < 4; i++) {

        dma_irqs[i] = qdev_get_gpio_in(s->ih[omap2_dma_irq_map[i].ih],

                                       omap2_dma_irq_map[i].intr);

    }

    s->dma = omap_dma4_init(0x48056000, dma_irqs, sysmem, s, 256, 32,

                    omap_findclk(s, "sdma_iclk"),

                    omap_findclk(s, "sdma_fclk"));

    s->port->addr_valid = omap2_validate_addr;



    /* Register SDRAM and SRAM ports for fast DMA transfers.  */

    soc_dma_port_add_mem(s->dma, memory_region_get_ram_ptr(&s->sdram),

                         OMAP2_Q2_BASE, s->sdram_size);

    soc_dma_port_add_mem(s->dma, memory_region_get_ram_ptr(&s->sram),

                         OMAP2_SRAM_BASE, s->sram_size);



    s->uart[0] = omap2_uart_init(sysmem, omap_l4ta(s->l4, 19),

                                 qdev_get_gpio_in(s->ih[0],

                                                  OMAP_INT_24XX_UART1_IRQ),

                    omap_findclk(s, "uart1_fclk"),

                    omap_findclk(s, "uart1_iclk"),

                    s->drq[OMAP24XX_DMA_UART1_TX],

                    s->drq[OMAP24XX_DMA_UART1_RX],

                    "uart1",

                    serial_hds[0]);

    s->uart[1] = omap2_uart_init(sysmem, omap_l4ta(s->l4, 20),

                                 qdev_get_gpio_in(s->ih[0],

                                                  OMAP_INT_24XX_UART2_IRQ),

                    omap_findclk(s, "uart2_fclk"),

                    omap_findclk(s, "uart2_iclk"),

                    s->drq[OMAP24XX_DMA_UART2_TX],

                    s->drq[OMAP24XX_DMA_UART2_RX],

                    "uart2",

                    serial_hds[0] ? serial_hds[1] : NULL);

    s->uart[2] = omap2_uart_init(sysmem, omap_l4ta(s->l4, 21),

                                 qdev_get_gpio_in(s->ih[0],

                                                  OMAP_INT_24XX_UART3_IRQ),

                    omap_findclk(s, "uart3_fclk"),

                    omap_findclk(s, "uart3_iclk"),

                    s->drq[OMAP24XX_DMA_UART3_TX],

                    s->drq[OMAP24XX_DMA_UART3_RX],

                    "uart3",

                    serial_hds[0] && serial_hds[1] ? serial_hds[2] : NULL);



    s->gptimer[0] = omap_gp_timer_init(omap_l4ta(s->l4, 7),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER1),

                    omap_findclk(s, "wu_gpt1_clk"),

                    omap_findclk(s, "wu_l4_iclk"));

    s->gptimer[1] = omap_gp_timer_init(omap_l4ta(s->l4, 8),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER2),

                    omap_findclk(s, "core_gpt2_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[2] = omap_gp_timer_init(omap_l4ta(s->l4, 22),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER3),

                    omap_findclk(s, "core_gpt3_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[3] = omap_gp_timer_init(omap_l4ta(s->l4, 23),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER4),

                    omap_findclk(s, "core_gpt4_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[4] = omap_gp_timer_init(omap_l4ta(s->l4, 24),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER5),

                    omap_findclk(s, "core_gpt5_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[5] = omap_gp_timer_init(omap_l4ta(s->l4, 25),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER6),

                    omap_findclk(s, "core_gpt6_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[6] = omap_gp_timer_init(omap_l4ta(s->l4, 26),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER7),

                    omap_findclk(s, "core_gpt7_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[7] = omap_gp_timer_init(omap_l4ta(s->l4, 27),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER8),

                    omap_findclk(s, "core_gpt8_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[8] = omap_gp_timer_init(omap_l4ta(s->l4, 28),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER9),

                    omap_findclk(s, "core_gpt9_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[9] = omap_gp_timer_init(omap_l4ta(s->l4, 29),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER10),

                    omap_findclk(s, "core_gpt10_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[10] = omap_gp_timer_init(omap_l4ta(s->l4, 30),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER11),

                    omap_findclk(s, "core_gpt11_clk"),

                    omap_findclk(s, "core_l4_iclk"));

    s->gptimer[11] = omap_gp_timer_init(omap_l4ta(s->l4, 31),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPTIMER12),

                    omap_findclk(s, "core_gpt12_clk"),

                    omap_findclk(s, "core_l4_iclk"));



    omap_tap_init(omap_l4ta(s->l4, 2), s);



    s->synctimer = omap_synctimer_init(omap_l4tao(s->l4, 2), s,

                    omap_findclk(s, "clk32-kHz"),

                    omap_findclk(s, "core_l4_iclk"));



    s->i2c[0] = qdev_create(NULL, "omap_i2c");

    qdev_prop_set_uint8(s->i2c[0], "revision", 0x34);

    qdev_prop_set_ptr(s->i2c[0], "iclk", omap_findclk(s, "i2c1.iclk"));

    qdev_prop_set_ptr(s->i2c[0], "fclk", omap_findclk(s, "i2c1.fclk"));

    qdev_init_nofail(s->i2c[0]);

    busdev = SYS_BUS_DEVICE(s->i2c[0]);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_I2C1_IRQ));

    sysbus_connect_irq(busdev, 1, s->drq[OMAP24XX_DMA_I2C1_TX]);

    sysbus_connect_irq(busdev, 2, s->drq[OMAP24XX_DMA_I2C1_RX]);

    sysbus_mmio_map(busdev, 0, omap_l4_region_base(omap_l4tao(s->l4, 5), 0));



    s->i2c[1] = qdev_create(NULL, "omap_i2c");

    qdev_prop_set_uint8(s->i2c[1], "revision", 0x34);

    qdev_prop_set_ptr(s->i2c[1], "iclk", omap_findclk(s, "i2c2.iclk"));

    qdev_prop_set_ptr(s->i2c[1], "fclk", omap_findclk(s, "i2c2.fclk"));

    qdev_init_nofail(s->i2c[1]);

    busdev = SYS_BUS_DEVICE(s->i2c[1]);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_I2C2_IRQ));

    sysbus_connect_irq(busdev, 1, s->drq[OMAP24XX_DMA_I2C2_TX]);

    sysbus_connect_irq(busdev, 2, s->drq[OMAP24XX_DMA_I2C2_RX]);

    sysbus_mmio_map(busdev, 0, omap_l4_region_base(omap_l4tao(s->l4, 6), 0));



    s->gpio = qdev_create(NULL, "omap2-gpio");

    qdev_prop_set_int32(s->gpio, "mpu_model", s->mpu_model);

    qdev_prop_set_ptr(s->gpio, "iclk", omap_findclk(s, "gpio_iclk"));

    qdev_prop_set_ptr(s->gpio, "fclk0", omap_findclk(s, "gpio1_dbclk"));

    qdev_prop_set_ptr(s->gpio, "fclk1", omap_findclk(s, "gpio2_dbclk"));

    qdev_prop_set_ptr(s->gpio, "fclk2", omap_findclk(s, "gpio3_dbclk"));

    qdev_prop_set_ptr(s->gpio, "fclk3", omap_findclk(s, "gpio4_dbclk"));

    if (s->mpu_model == omap2430) {

        qdev_prop_set_ptr(s->gpio, "fclk4", omap_findclk(s, "gpio5_dbclk"));

    }

    qdev_init_nofail(s->gpio);

    busdev = SYS_BUS_DEVICE(s->gpio);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPIO_BANK1));

    sysbus_connect_irq(busdev, 3,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPIO_BANK2));

    sysbus_connect_irq(busdev, 6,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPIO_BANK3));

    sysbus_connect_irq(busdev, 9,

                       qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPIO_BANK4));

    if (s->mpu_model == omap2430) {

        sysbus_connect_irq(busdev, 12,

                           qdev_get_gpio_in(s->ih[0],

                                            OMAP_INT_243X_GPIO_BANK5));

    }

    ta = omap_l4ta(s->l4, 3);

    sysbus_mmio_map(busdev, 0, omap_l4_region_base(ta, 1));

    sysbus_mmio_map(busdev, 1, omap_l4_region_base(ta, 0));

    sysbus_mmio_map(busdev, 2, omap_l4_region_base(ta, 2));

    sysbus_mmio_map(busdev, 3, omap_l4_region_base(ta, 4));

    sysbus_mmio_map(busdev, 4, omap_l4_region_base(ta, 5));



    s->sdrc = omap_sdrc_init(sysmem, 0x68009000);

    s->gpmc = omap_gpmc_init(s, 0x6800a000,

                             qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_GPMC_IRQ),

                             s->drq[OMAP24XX_DMA_GPMC]);



    dinfo = drive_get(IF_SD, 0, 0);

    if (!dinfo) {

        fprintf(stderr, "qemu: missing SecureDigital device\n");

        exit(1);

    }

    s->mmc = omap2_mmc_init(omap_l4tao(s->l4, 9),

                    blk_bs(blk_by_legacy_dinfo(dinfo)),

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_MMC_IRQ),

                    &s->drq[OMAP24XX_DMA_MMC1_TX],

                    omap_findclk(s, "mmc_fclk"), omap_findclk(s, "mmc_iclk"));



    s->mcspi[0] = omap_mcspi_init(omap_l4ta(s->l4, 35), 4,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_MCSPI1_IRQ),

                    &s->drq[OMAP24XX_DMA_SPI1_TX0],

                    omap_findclk(s, "spi1_fclk"),

                    omap_findclk(s, "spi1_iclk"));

    s->mcspi[1] = omap_mcspi_init(omap_l4ta(s->l4, 36), 2,

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_MCSPI2_IRQ),

                    &s->drq[OMAP24XX_DMA_SPI2_TX0],

                    omap_findclk(s, "spi2_fclk"),

                    omap_findclk(s, "spi2_iclk"));



    s->dss = omap_dss_init(omap_l4ta(s->l4, 10), sysmem, 0x68000800,

                    /* XXX wire M_IRQ_25, D_L2_IRQ_30 and I_IRQ_13 together */

                    qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_DSS_IRQ),

                           s->drq[OMAP24XX_DMA_DSS],

                    omap_findclk(s, "dss_clk1"), omap_findclk(s, "dss_clk2"),

                    omap_findclk(s, "dss_54m_clk"),

                    omap_findclk(s, "dss_l3_iclk"),

                    omap_findclk(s, "dss_l4_iclk"));



    omap_sti_init(omap_l4ta(s->l4, 18), sysmem, 0x54000000,

                  qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_STI),

                  omap_findclk(s, "emul_ck"),

                    serial_hds[0] && serial_hds[1] && serial_hds[2] ?

                    serial_hds[3] : NULL);



    s->eac = omap_eac_init(omap_l4ta(s->l4, 32),

                           qdev_get_gpio_in(s->ih[0], OMAP_INT_24XX_EAC_IRQ),

                    /* Ten consecutive lines */

                    &s->drq[OMAP24XX_DMA_EAC_AC_RD],

                    omap_findclk(s, "func_96m_clk"),

                    omap_findclk(s, "core_l4_iclk"));



    /* All register mappings (includin those not currenlty implemented):

     * SystemControlMod	48000000 - 48000fff

     * SystemControlL4	48001000 - 48001fff

     * 32kHz Timer Mod	48004000 - 48004fff

     * 32kHz Timer L4	48005000 - 48005fff

     * PRCM ModA	48008000 - 480087ff

     * PRCM ModB	48008800 - 48008fff

     * PRCM L4		48009000 - 48009fff

     * TEST-BCM Mod	48012000 - 48012fff

     * TEST-BCM L4	48013000 - 48013fff

     * TEST-TAP Mod	48014000 - 48014fff

     * TEST-TAP L4	48015000 - 48015fff

     * GPIO1 Mod	48018000 - 48018fff

     * GPIO Top		48019000 - 48019fff

     * GPIO2 Mod	4801a000 - 4801afff

     * GPIO L4		4801b000 - 4801bfff

     * GPIO3 Mod	4801c000 - 4801cfff

     * GPIO4 Mod	4801e000 - 4801efff

     * WDTIMER1 Mod	48020000 - 48010fff

     * WDTIMER Top	48021000 - 48011fff

     * WDTIMER2 Mod	48022000 - 48012fff

     * WDTIMER L4	48023000 - 48013fff

     * WDTIMER3 Mod	48024000 - 48014fff

     * WDTIMER3 L4	48025000 - 48015fff

     * WDTIMER4 Mod	48026000 - 48016fff

     * WDTIMER4 L4	48027000 - 48017fff

     * GPTIMER1 Mod	48028000 - 48018fff

     * GPTIMER1 L4	48029000 - 48019fff

     * GPTIMER2 Mod	4802a000 - 4801afff

     * GPTIMER2 L4	4802b000 - 4801bfff

     * L4-Config AP	48040000 - 480407ff

     * L4-Config IP	48040800 - 48040fff

     * L4-Config LA	48041000 - 48041fff

     * ARM11ETB Mod	48048000 - 48049fff

     * ARM11ETB L4	4804a000 - 4804afff

     * DISPLAY Top	48050000 - 480503ff

     * DISPLAY DISPC	48050400 - 480507ff

     * DISPLAY RFBI	48050800 - 48050bff

     * DISPLAY VENC	48050c00 - 48050fff

     * DISPLAY L4	48051000 - 48051fff

     * CAMERA Top	48052000 - 480523ff

     * CAMERA core	48052400 - 480527ff

     * CAMERA DMA	48052800 - 48052bff

     * CAMERA MMU	48052c00 - 48052fff

     * CAMERA L4	48053000 - 48053fff

     * SDMA Mod		48056000 - 48056fff

     * SDMA L4		48057000 - 48057fff

     * SSI Top		48058000 - 48058fff

     * SSI GDD		48059000 - 48059fff

     * SSI Port1	4805a000 - 4805afff

     * SSI Port2	4805b000 - 4805bfff

     * SSI L4		4805c000 - 4805cfff

     * USB Mod		4805e000 - 480fefff

     * USB L4		4805f000 - 480fffff

     * WIN_TRACER1 Mod	48060000 - 48060fff

     * WIN_TRACER1 L4	48061000 - 48061fff

     * WIN_TRACER2 Mod	48062000 - 48062fff

     * WIN_TRACER2 L4	48063000 - 48063fff

     * WIN_TRACER3 Mod	48064000 - 48064fff

     * WIN_TRACER3 L4	48065000 - 48065fff

     * WIN_TRACER4 Top	48066000 - 480660ff

     * WIN_TRACER4 ETT	48066100 - 480661ff

     * WIN_TRACER4 WT	48066200 - 480662ff

     * WIN_TRACER4 L4	48067000 - 48067fff

     * XTI Mod		48068000 - 48068fff

     * XTI L4		48069000 - 48069fff

     * UART1 Mod	4806a000 - 4806afff

     * UART1 L4		4806b000 - 4806bfff

     * UART2 Mod	4806c000 - 4806cfff

     * UART2 L4		4806d000 - 4806dfff

     * UART3 Mod	4806e000 - 4806efff

     * UART3 L4		4806f000 - 4806ffff

     * I2C1 Mod		48070000 - 48070fff

     * I2C1 L4		48071000 - 48071fff

     * I2C2 Mod		48072000 - 48072fff

     * I2C2 L4		48073000 - 48073fff

     * McBSP1 Mod	48074000 - 48074fff

     * McBSP1 L4	48075000 - 48075fff

     * McBSP2 Mod	48076000 - 48076fff

     * McBSP2 L4	48077000 - 48077fff

     * GPTIMER3 Mod	48078000 - 48078fff

     * GPTIMER3 L4	48079000 - 48079fff

     * GPTIMER4 Mod	4807a000 - 4807afff

     * GPTIMER4 L4	4807b000 - 4807bfff

     * GPTIMER5 Mod	4807c000 - 4807cfff

     * GPTIMER5 L4	4807d000 - 4807dfff

     * GPTIMER6 Mod	4807e000 - 4807efff

     * GPTIMER6 L4	4807f000 - 4807ffff

     * GPTIMER7 Mod	48080000 - 48080fff

     * GPTIMER7 L4	48081000 - 48081fff

     * GPTIMER8 Mod	48082000 - 48082fff

     * GPTIMER8 L4	48083000 - 48083fff

     * GPTIMER9 Mod	48084000 - 48084fff

     * GPTIMER9 L4	48085000 - 48085fff

     * GPTIMER10 Mod	48086000 - 48086fff

     * GPTIMER10 L4	48087000 - 48087fff

     * GPTIMER11 Mod	48088000 - 48088fff

     * GPTIMER11 L4	48089000 - 48089fff

     * GPTIMER12 Mod	4808a000 - 4808afff

     * GPTIMER12 L4	4808b000 - 4808bfff

     * EAC Mod		48090000 - 48090fff

     * EAC L4		48091000 - 48091fff

     * FAC Mod		48092000 - 48092fff

     * FAC L4		48093000 - 48093fff

     * MAILBOX Mod	48094000 - 48094fff

     * MAILBOX L4	48095000 - 48095fff

     * SPI1 Mod		48098000 - 48098fff

     * SPI1 L4		48099000 - 48099fff

     * SPI2 Mod		4809a000 - 4809afff

     * SPI2 L4		4809b000 - 4809bfff

     * MMC/SDIO Mod	4809c000 - 4809cfff

     * MMC/SDIO L4	4809d000 - 4809dfff

     * MS_PRO Mod	4809e000 - 4809efff

     * MS_PRO L4	4809f000 - 4809ffff

     * RNG Mod		480a0000 - 480a0fff

     * RNG L4		480a1000 - 480a1fff

     * DES3DES Mod	480a2000 - 480a2fff

     * DES3DES L4	480a3000 - 480a3fff

     * SHA1MD5 Mod	480a4000 - 480a4fff

     * SHA1MD5 L4	480a5000 - 480a5fff

     * AES Mod		480a6000 - 480a6fff

     * AES L4		480a7000 - 480a7fff

     * PKA Mod		480a8000 - 480a9fff

     * PKA L4		480aa000 - 480aafff

     * MG Mod		480b0000 - 480b0fff

     * MG L4		480b1000 - 480b1fff

     * HDQ/1-wire Mod	480b2000 - 480b2fff

     * HDQ/1-wire L4	480b3000 - 480b3fff

     * MPU interrupt	480fe000 - 480fefff

     * STI channel base	54000000 - 5400ffff

     * IVA RAM		5c000000 - 5c01ffff

     * IVA ROM		5c020000 - 5c027fff

     * IMG_BUF_A	5c040000 - 5c040fff

     * IMG_BUF_B	5c042000 - 5c042fff

     * VLCDS		5c048000 - 5c0487ff

     * IMX_COEF		5c049000 - 5c04afff

     * IMX_CMD		5c051000 - 5c051fff

     * VLCDQ		5c053000 - 5c0533ff

     * VLCDH		5c054000 - 5c054fff

     * SEQ_CMD		5c055000 - 5c055fff

     * IMX_REG		5c056000 - 5c0560ff

     * VLCD_REG		5c056100 - 5c0561ff

     * SEQ_REG		5c056200 - 5c0562ff

     * IMG_BUF_REG	5c056300 - 5c0563ff

     * SEQIRQ_REG	5c056400 - 5c0564ff

     * OCP_REG		5c060000 - 5c060fff

     * SYSC_REG		5c070000 - 5c070fff

     * MMU_REG		5d000000 - 5d000fff

     * sDMA R		68000400 - 680005ff

     * sDMA W		68000600 - 680007ff

     * Display Control	68000800 - 680009ff

     * DSP subsystem	68000a00 - 68000bff

     * MPU subsystem	68000c00 - 68000dff

     * IVA subsystem	68001000 - 680011ff

     * USB		68001200 - 680013ff

     * Camera		68001400 - 680015ff

     * VLYNQ (firewall)	68001800 - 68001bff

     * VLYNQ		68001e00 - 68001fff

     * SSI		68002000 - 680021ff

     * L4		68002400 - 680025ff

     * DSP (firewall)	68002800 - 68002bff

     * DSP subsystem	68002e00 - 68002fff

     * IVA (firewall)	68003000 - 680033ff

     * IVA		68003600 - 680037ff

     * GFX		68003a00 - 68003bff

     * CMDWR emulation	68003c00 - 68003dff

     * SMS		68004000 - 680041ff

     * OCM		68004200 - 680043ff

     * GPMC		68004400 - 680045ff

     * RAM (firewall)	68005000 - 680053ff

     * RAM (err login)	68005400 - 680057ff

     * ROM (firewall)	68005800 - 68005bff

     * ROM (err login)	68005c00 - 68005fff

     * GPMC (firewall)	68006000 - 680063ff

     * GPMC (err login)	68006400 - 680067ff

     * SMS (err login)	68006c00 - 68006fff

     * SMS registers	68008000 - 68008fff

     * SDRC registers	68009000 - 68009fff

     * GPMC registers	6800a000   6800afff

     */



    qemu_register_reset(omap2_mpu_reset, s);



    return s;

}
