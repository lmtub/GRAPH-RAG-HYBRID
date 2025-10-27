PXA2xxState *pxa255_init(MemoryRegion *address_space, unsigned int sdram_size)

{

    PXA2xxState *s;

    int i;

    DriveInfo *dinfo;



    s = (PXA2xxState *) g_malloc0(sizeof(PXA2xxState));



    s->cpu = cpu_arm_init("pxa255");

    if (s->cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    s->reset = qemu_allocate_irq(pxa2xx_reset, s, 0);



    /* SDRAM & Internal Memory Storage */

    memory_region_init_ram(&s->sdram, NULL, "pxa255.sdram", sdram_size,

                           &error_abort);

    vmstate_register_ram_global(&s->sdram);

    memory_region_add_subregion(address_space, PXA2XX_SDRAM_BASE, &s->sdram);

    memory_region_init_ram(&s->internal, NULL, "pxa255.internal",

                           PXA2XX_INTERNAL_SIZE, &error_abort);

    vmstate_register_ram_global(&s->internal);

    memory_region_add_subregion(address_space, PXA2XX_INTERNAL_BASE,

                                &s->internal);



    s->pic = pxa2xx_pic_init(0x40d00000, s->cpu);



    s->dma = pxa255_dma_init(0x40000000,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_DMA));



    sysbus_create_varargs("pxa25x-timer", 0x40a00000,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_OST_0 + 0),

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_OST_0 + 1),

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_OST_0 + 2),

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_OST_0 + 3),

                    NULL);



    s->gpio = pxa2xx_gpio_init(0x40e00000, s->cpu, s->pic, 85);



    dinfo = drive_get(IF_SD, 0, 0);

    if (!dinfo) {

        fprintf(stderr, "qemu: missing SecureDigital device\n");

        exit(1);

    }

    s->mmc = pxa2xx_mmci_init(address_space, 0x41100000,

                    blk_by_legacy_dinfo(dinfo),

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_MMC),

                    qdev_get_gpio_in(s->dma, PXA2XX_RX_RQ_MMCI),

                    qdev_get_gpio_in(s->dma, PXA2XX_TX_RQ_MMCI));



    for (i = 0; pxa255_serial[i].io_base; i++) {

        if (serial_hds[i]) {

            serial_mm_init(address_space, pxa255_serial[i].io_base, 2,

                           qdev_get_gpio_in(s->pic, pxa255_serial[i].irqn),

                           14745600 / 16, serial_hds[i],

                           DEVICE_NATIVE_ENDIAN);

        } else {

            break;

        }

    }

    if (serial_hds[i])

        s->fir = pxa2xx_fir_init(address_space, 0x40800000,

                        qdev_get_gpio_in(s->pic, PXA2XX_PIC_ICP),

                        qdev_get_gpio_in(s->dma, PXA2XX_RX_RQ_ICP),

                        qdev_get_gpio_in(s->dma, PXA2XX_TX_RQ_ICP),

                        serial_hds[i]);



    s->lcd = pxa2xx_lcdc_init(address_space, 0x44000000,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_LCD));



    s->cm_base = 0x41300000;

    s->cm_regs[CCCR >> 2] = 0x02000210;	/* 416.0 MHz */

    s->clkcfg = 0x00000009;		/* Turbo mode active */

    memory_region_init_io(&s->cm_iomem, NULL, &pxa2xx_cm_ops, s, "pxa2xx-cm", 0x1000);

    memory_region_add_subregion(address_space, s->cm_base, &s->cm_iomem);

    vmstate_register(NULL, 0, &vmstate_pxa2xx_cm, s);



    pxa2xx_setup_cp14(s);



    s->mm_base = 0x48000000;

    s->mm_regs[MDMRS >> 2] = 0x00020002;

    s->mm_regs[MDREFR >> 2] = 0x03ca4000;

    s->mm_regs[MECR >> 2] = 0x00000001;	/* Two PC Card sockets */

    memory_region_init_io(&s->mm_iomem, NULL, &pxa2xx_mm_ops, s, "pxa2xx-mm", 0x1000);

    memory_region_add_subregion(address_space, s->mm_base, &s->mm_iomem);

    vmstate_register(NULL, 0, &vmstate_pxa2xx_mm, s);



    s->pm_base = 0x40f00000;

    memory_region_init_io(&s->pm_iomem, NULL, &pxa2xx_pm_ops, s, "pxa2xx-pm", 0x100);

    memory_region_add_subregion(address_space, s->pm_base, &s->pm_iomem);

    vmstate_register(NULL, 0, &vmstate_pxa2xx_pm, s);



    for (i = 0; pxa255_ssp[i].io_base; i ++);

    s->ssp = (SSIBus **)g_malloc0(sizeof(SSIBus *) * i);

    for (i = 0; pxa255_ssp[i].io_base; i ++) {

        DeviceState *dev;

        dev = sysbus_create_simple(TYPE_PXA2XX_SSP, pxa255_ssp[i].io_base,

                        qdev_get_gpio_in(s->pic, pxa255_ssp[i].irqn));

        s->ssp[i] = (SSIBus *)qdev_get_child_bus(dev, "ssi");

    }



    if (usb_enabled()) {

        sysbus_create_simple("sysbus-ohci", 0x4c000000,

                        qdev_get_gpio_in(s->pic, PXA2XX_PIC_USBH1));

    }



    s->pcmcia[0] = pxa2xx_pcmcia_init(address_space, 0x20000000);

    s->pcmcia[1] = pxa2xx_pcmcia_init(address_space, 0x30000000);



    sysbus_create_simple(TYPE_PXA2XX_RTC, 0x40900000,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_RTCALARM));



    s->i2c[0] = pxa2xx_i2c_init(0x40301600,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_I2C), 0xffff);

    s->i2c[1] = pxa2xx_i2c_init(0x40f00100,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_PWRI2C), 0xff);



    s->i2s = pxa2xx_i2s_init(address_space, 0x40400000,

                    qdev_get_gpio_in(s->pic, PXA2XX_PIC_I2S),

                    qdev_get_gpio_in(s->dma, PXA2XX_RX_RQ_I2S),

                    qdev_get_gpio_in(s->dma, PXA2XX_TX_RQ_I2S));



    /* GPIO1 resets the processor */

    /* The handler can be overridden by board-specific code */

    qdev_connect_gpio_out(s->gpio, 1, s->reset);

    return s;

}
