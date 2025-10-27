PXA2xxState *pxa270_init(unsigned int sdram_size, const char *revision)

{

    PXA2xxState *s;

    int iomemtype, i;

    DriveInfo *dinfo;

    s = (PXA2xxState *) qemu_mallocz(sizeof(PXA2xxState));



    if (revision && strncmp(revision, "pxa27", 5)) {

        fprintf(stderr, "Machine requires a PXA27x processor.\n");

        exit(1);

    }

    if (!revision)

        revision = "pxa270";

    

    s->env = cpu_init(revision);

    if (!s->env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    s->reset = qemu_allocate_irqs(pxa2xx_reset, s, 1)[0];



    /* SDRAM & Internal Memory Storage */

    cpu_register_physical_memory(PXA2XX_SDRAM_BASE,

                    sdram_size, qemu_ram_alloc(NULL, "pxa270.sdram",

                                               sdram_size) | IO_MEM_RAM);

    cpu_register_physical_memory(PXA2XX_INTERNAL_BASE,

                    0x40000, qemu_ram_alloc(NULL, "pxa270.internal",

                                            0x40000) | IO_MEM_RAM);



    s->pic = pxa2xx_pic_init(0x40d00000, s->env);



    s->dma = pxa27x_dma_init(0x40000000, s->pic[PXA2XX_PIC_DMA]);



    pxa27x_timer_init(0x40a00000, &s->pic[PXA2XX_PIC_OST_0],

                    s->pic[PXA27X_PIC_OST_4_11]);



    s->gpio = pxa2xx_gpio_init(0x40e00000, s->env, s->pic, 121);



    dinfo = drive_get(IF_SD, 0, 0);

    if (!dinfo) {

        fprintf(stderr, "qemu: missing SecureDigital device\n");

        exit(1);

    }

    s->mmc = pxa2xx_mmci_init(0x41100000, dinfo->bdrv,

                              s->pic[PXA2XX_PIC_MMC], s->dma);



    for (i = 0; pxa270_serial[i].io_base; i ++)

        if (serial_hds[i])

#ifdef TARGET_WORDS_BIGENDIAN

            serial_mm_init(pxa270_serial[i].io_base, 2,

                           s->pic[pxa270_serial[i].irqn], 14857000/16,

                           serial_hds[i], 1, 1);

#else

            serial_mm_init(pxa270_serial[i].io_base, 2,

                           s->pic[pxa270_serial[i].irqn], 14857000/16,

                           serial_hds[i], 1, 0);

#endif

        else

            break;

    if (serial_hds[i])

        s->fir = pxa2xx_fir_init(0x40800000, s->pic[PXA2XX_PIC_ICP],

                        s->dma, serial_hds[i]);



    s->lcd = pxa2xx_lcdc_init(0x44000000, s->pic[PXA2XX_PIC_LCD]);



    s->cm_base = 0x41300000;

    s->cm_regs[CCCR >> 2] = 0x02000210;	/* 416.0 MHz */

    s->clkcfg = 0x00000009;		/* Turbo mode active */

    iomemtype = cpu_register_io_memory(pxa2xx_cm_readfn,

                    pxa2xx_cm_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(s->cm_base, 0x1000, iomemtype);

    register_savevm(NULL, "pxa2xx_cm", 0, 0, pxa2xx_cm_save, pxa2xx_cm_load, s);



    cpu_arm_set_cp_io(s->env, 14, pxa2xx_cp14_read, pxa2xx_cp14_write, s);



    s->mm_base = 0x48000000;

    s->mm_regs[MDMRS >> 2] = 0x00020002;

    s->mm_regs[MDREFR >> 2] = 0x03ca4000;

    s->mm_regs[MECR >> 2] = 0x00000001;	/* Two PC Card sockets */

    iomemtype = cpu_register_io_memory(pxa2xx_mm_readfn,

                    pxa2xx_mm_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(s->mm_base, 0x1000, iomemtype);

    register_savevm(NULL, "pxa2xx_mm", 0, 0, pxa2xx_mm_save, pxa2xx_mm_load, s);



    s->pm_base = 0x40f00000;

    iomemtype = cpu_register_io_memory(pxa2xx_pm_readfn,

                    pxa2xx_pm_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(s->pm_base, 0x100, iomemtype);

    register_savevm(NULL, "pxa2xx_pm", 0, 0, pxa2xx_pm_save, pxa2xx_pm_load, s);



    for (i = 0; pxa27x_ssp[i].io_base; i ++);

    s->ssp = (SSIBus **)qemu_mallocz(sizeof(SSIBus *) * i);

    for (i = 0; pxa27x_ssp[i].io_base; i ++) {

        DeviceState *dev;

        dev = sysbus_create_simple("pxa2xx-ssp", pxa27x_ssp[i].io_base,

                                   s->pic[pxa27x_ssp[i].irqn]);

        s->ssp[i] = (SSIBus *)qdev_get_child_bus(dev, "ssi");

    }



    if (usb_enabled) {

        sysbus_create_simple("sysbus-ohci", 0x4c000000,

                             s->pic[PXA2XX_PIC_USBH1]);

    }



    s->pcmcia[0] = pxa2xx_pcmcia_init(0x20000000);

    s->pcmcia[1] = pxa2xx_pcmcia_init(0x30000000);



    s->rtc_base = 0x40900000;

    iomemtype = cpu_register_io_memory(pxa2xx_rtc_readfn,

                    pxa2xx_rtc_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(s->rtc_base, 0x1000, iomemtype);

    pxa2xx_rtc_init(s);

    register_savevm(NULL, "pxa2xx_rtc", 0, 0, pxa2xx_rtc_save,

                    pxa2xx_rtc_load, s);



    s->i2c[0] = pxa2xx_i2c_init(0x40301600, s->pic[PXA2XX_PIC_I2C], 0xffff);

    s->i2c[1] = pxa2xx_i2c_init(0x40f00100, s->pic[PXA2XX_PIC_PWRI2C], 0xff);



    s->i2s = pxa2xx_i2s_init(0x40400000, s->pic[PXA2XX_PIC_I2S], s->dma);



    s->kp = pxa27x_keypad_init(0x41500000, s->pic[PXA2XX_PIC_KEYPAD]);



    /* GPIO1 resets the processor */

    /* The handler can be overridden by board-specific code */

    qdev_connect_gpio_out(s->gpio, 1, s->reset);

    return s;

}
