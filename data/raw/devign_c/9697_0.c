static void n8x0_init(ram_addr_t ram_size, const char *boot_device,

                const char *kernel_filename,

                const char *kernel_cmdline, const char *initrd_filename,

                const char *cpu_model, struct arm_boot_info *binfo, int model)

{

    struct n800_s *s = (struct n800_s *) qemu_mallocz(sizeof(*s));

    int sdram_size = binfo->ram_size;

    int onenandram_size = 0x00010000;

    DisplayState *ds;



    if (ram_size < sdram_size + onenandram_size + OMAP242X_SRAM_SIZE) {

        fprintf(stderr, "This architecture uses %i bytes of memory\n",

                        sdram_size + onenandram_size + OMAP242X_SRAM_SIZE);

        exit(1);

    }



    s->cpu = omap2420_mpu_init(sdram_size, cpu_model);



    /* Setup peripherals

     *

     * Believed external peripherals layout in the N810:

     * (spi bus 1)

     *   tsc2005

     *   lcd_mipid

     * (spi bus 2)

     *   Conexant cx3110x (WLAN)

     *   optional: pc2400m (WiMAX)

     * (i2c bus 0)

     *   TLV320AIC33 (audio codec)

     *   TCM825x (camera by Toshiba)

     *   lp5521 (clever LEDs)

     *   tsl2563 (light sensor, hwmon, model 7, rev. 0)

     *   lm8323 (keypad, manf 00, rev 04)

     * (i2c bus 1)

     *   tmp105 (temperature sensor, hwmon)

     *   menelaus (pm)

     * (somewhere on i2c - maybe N800-only)

     *   tea5761 (FM tuner)

     * (serial 0)

     *   GPS

     * (some serial port)

     *   csr41814 (Bluetooth)

     */

    n8x0_gpio_setup(s);

    n8x0_nand_setup(s);

    n8x0_i2c_setup(s);

    if (model == 800)

        n800_tsc_kbd_setup(s);

    else if (model == 810) {

        n810_tsc_setup(s);

        n810_kbd_setup(s);

    }

    n8x0_spi_setup(s);

    n8x0_dss_setup(s);

    n8x0_cbus_setup(s);

    n8x0_uart_setup(s);

    if (usb_enabled)

        n8x0_usb_setup(s);



    /* Setup initial (reset) machine state */



    /* Start at the OneNAND bootloader.  */

    s->cpu->env->regs[15] = 0;



    if (kernel_filename) {

        /* Or at the linux loader.  */

        binfo->kernel_filename = kernel_filename;

        binfo->kernel_cmdline = kernel_cmdline;

        binfo->initrd_filename = initrd_filename;

        arm_load_kernel(s->cpu->env, binfo);



        qemu_register_reset(n8x0_boot_init, s);

        n8x0_boot_init(s);

    }



    if (option_rom[0] && (boot_device[0] == 'n' || !kernel_filename)) {

        int rom_size;

        uint8_t nolo_tags[0x10000];

        /* No, wait, better start at the ROM.  */

        s->cpu->env->regs[15] = OMAP2_Q2_BASE + 0x400000;



        /* This is intended for loading the `secondary.bin' program from

         * Nokia images (the NOLO bootloader).  The entry point seems

         * to be at OMAP2_Q2_BASE + 0x400000.

         *

         * The `2nd.bin' files contain some kind of earlier boot code and

         * for them the entry point needs to be set to OMAP2_SRAM_BASE.

         *

         * The code above is for loading the `zImage' file from Nokia

         * images.  */

        rom_size = load_image_targphys(option_rom[0],

                                       OMAP2_Q2_BASE + 0x400000,

                                       sdram_size - 0x400000);

        printf("%i bytes of image loaded\n", rom_size);



        n800_setup_nolo_tags(nolo_tags);

        cpu_physical_memory_write(OMAP2_SRAM_BASE, nolo_tags, 0x10000);

    }

    /* FIXME: We shouldn't really be doing this here.  The LCD controller

       will set the size once configured, so this just sets an initial

       size until the guest activates the display.  */

    ds = get_displaystate();

    ds->surface = qemu_resize_displaysurface(ds, 800, 480);

    dpy_resize(ds);

}
