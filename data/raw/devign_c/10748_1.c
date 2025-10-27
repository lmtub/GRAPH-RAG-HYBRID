static void z2_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    MemoryRegion *address_space_mem = get_system_memory();

    uint32_t sector_len = 0x10000;

    PXA2xxState *mpu;

    DriveInfo *dinfo;

    int be;

    void *z2_lcd;

    I2CBus *bus;

    DeviceState *wm;



    if (!cpu_model) {

        cpu_model = "pxa270-c5";

    }



    /* Setup CPU & memory */

    mpu = pxa270_init(address_space_mem, z2_binfo.ram_size, cpu_model);



#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    dinfo = drive_get(IF_PFLASH, 0, 0);

    if (!dinfo && !qtest_enabled()) {

        fprintf(stderr, "Flash image must be given with the "

                "'pflash' parameter\n");

        exit(1);

    }



    if (!pflash_cfi01_register(Z2_FLASH_BASE,

                               NULL, "z2.flash0", Z2_FLASH_SIZE,

                               dinfo ? dinfo->bdrv : NULL, sector_len,

                               Z2_FLASH_SIZE / sector_len, 4, 0, 0, 0, 0,

                               be)) {

        fprintf(stderr, "qemu: Error registering flash memory.\n");

        exit(1);

    }



    /* setup keypad */

    pxa27x_register_keypad(mpu->kp, map, 0x100);



    /* MMC/SD host */

    pxa2xx_mmci_handlers(mpu->mmc,

        NULL,

        qdev_get_gpio_in(mpu->gpio, Z2_GPIO_SD_DETECT));



    type_register_static(&zipit_lcd_info);

    type_register_static(&aer915_info);

    z2_lcd = ssi_create_slave(mpu->ssp[1], "zipit-lcd");

    bus = pxa2xx_i2c_bus(mpu->i2c[0]);

    i2c_create_slave(bus, TYPE_AER915, 0x55);

    wm = i2c_create_slave(bus, "wm8750", 0x1b);

    mpu->i2s->opaque = wm;

    mpu->i2s->codec_out = wm8750_dac_dat;

    mpu->i2s->codec_in = wm8750_adc_dat;

    wm8750_data_req_set(wm, mpu->i2s->data_req, mpu->i2s);



    qdev_connect_gpio_out(mpu->gpio, Z2_GPIO_LCD_CS,

        qemu_allocate_irqs(z2_lcd_cs, z2_lcd, 1)[0]);



    z2_binfo.kernel_filename = kernel_filename;

    z2_binfo.kernel_cmdline = kernel_cmdline;

    z2_binfo.initrd_filename = initrd_filename;

    z2_binfo.board_id = 0x6dd;

    arm_load_kernel(mpu->cpu, &z2_binfo);

}
