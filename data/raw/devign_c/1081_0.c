static void palmte_init(MachineState *machine)

{

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    MemoryRegion *address_space_mem = get_system_memory();

    struct omap_mpu_state_s *mpu;

    int flash_size = 0x00800000;

    int sdram_size = palmte_binfo.ram_size;

    static uint32_t cs0val = 0xffffffff;

    static uint32_t cs1val = 0x0000e1a0;

    static uint32_t cs2val = 0x0000e1a0;

    static uint32_t cs3val = 0xe1a0e1a0;

    int rom_size, rom_loaded = 0;

    MemoryRegion *flash = g_new(MemoryRegion, 1);

    MemoryRegion *cs = g_new(MemoryRegion, 4);



    mpu = omap310_mpu_init(address_space_mem, sdram_size, machine->cpu_type);



    /* External Flash (EMIFS) */

    memory_region_init_ram(flash, NULL, "palmte.flash", flash_size,

                           &error_fatal);

    memory_region_set_readonly(flash, true);

    memory_region_add_subregion(address_space_mem, OMAP_CS0_BASE, flash);



    memory_region_init_io(&cs[0], NULL, &static_ops, &cs0val, "palmte-cs0",

                          OMAP_CS0_SIZE - flash_size);

    memory_region_add_subregion(address_space_mem, OMAP_CS0_BASE + flash_size,

                                &cs[0]);

    memory_region_init_io(&cs[1], NULL, &static_ops, &cs1val, "palmte-cs1",

                          OMAP_CS1_SIZE);

    memory_region_add_subregion(address_space_mem, OMAP_CS1_BASE, &cs[1]);

    memory_region_init_io(&cs[2], NULL, &static_ops, &cs2val, "palmte-cs2",

                          OMAP_CS2_SIZE);

    memory_region_add_subregion(address_space_mem, OMAP_CS2_BASE, &cs[2]);

    memory_region_init_io(&cs[3], NULL, &static_ops, &cs3val, "palmte-cs3",

                          OMAP_CS3_SIZE);

    memory_region_add_subregion(address_space_mem, OMAP_CS3_BASE, &cs[3]);



    palmte_microwire_setup(mpu);



    qemu_add_kbd_event_handler(palmte_button_event, mpu);



    palmte_gpio_setup(mpu);



    /* Setup initial (reset) machine state */

    if (nb_option_roms) {

        rom_size = get_image_size(option_rom[0].name);

        if (rom_size > flash_size) {

            fprintf(stderr, "%s: ROM image too big (%x > %x)\n",

                            __FUNCTION__, rom_size, flash_size);

            rom_size = 0;

        }

        if (rom_size > 0) {

            rom_size = load_image_targphys(option_rom[0].name, OMAP_CS0_BASE,

                                           flash_size);

            rom_loaded = 1;

        }

        if (rom_size < 0) {

            fprintf(stderr, "%s: error loading '%s'\n",

                            __FUNCTION__, option_rom[0].name);

        }

    }



    if (!rom_loaded && !kernel_filename && !qtest_enabled()) {

        fprintf(stderr, "Kernel or ROM image must be specified\n");

        exit(1);

    }



    /* Load the kernel.  */

    palmte_binfo.kernel_filename = kernel_filename;

    palmte_binfo.kernel_cmdline = kernel_cmdline;

    palmte_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(mpu->cpu, &palmte_binfo);

}
