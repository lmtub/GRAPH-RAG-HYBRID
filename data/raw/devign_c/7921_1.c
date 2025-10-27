static void sx1_init(MachineState *machine, const int version)

{

    struct omap_mpu_state_s *mpu;

    MemoryRegion *address_space = get_system_memory();

    MemoryRegion *flash = g_new(MemoryRegion, 1);

    MemoryRegion *flash_1 = g_new(MemoryRegion, 1);

    MemoryRegion *cs = g_new(MemoryRegion, 4);

    static uint32_t cs0val = 0x00213090;

    static uint32_t cs1val = 0x00215070;

    static uint32_t cs2val = 0x00001139;

    static uint32_t cs3val = 0x00001139;

    DriveInfo *dinfo;

    int fl_idx;

    uint32_t flash_size = flash0_size;

    int be;



    if (version == 2) {

        flash_size = flash2_size;

    }



    mpu = omap310_mpu_init(address_space, sx1_binfo.ram_size,

                           machine->cpu_model);



    /* External Flash (EMIFS) */

    memory_region_init_ram(flash, NULL, "omap_sx1.flash0-0", flash_size,

                           &error_abort);

    vmstate_register_ram_global(flash);

    memory_region_set_readonly(flash, true);

    memory_region_add_subregion(address_space, OMAP_CS0_BASE, flash);



    memory_region_init_io(&cs[0], NULL, &static_ops, &cs0val,

                          "sx1.cs0", OMAP_CS0_SIZE - flash_size);

    memory_region_add_subregion(address_space,

                                OMAP_CS0_BASE + flash_size, &cs[0]);





    memory_region_init_io(&cs[2], NULL, &static_ops, &cs2val,

                          "sx1.cs2", OMAP_CS2_SIZE);

    memory_region_add_subregion(address_space,

                                OMAP_CS2_BASE, &cs[2]);



    memory_region_init_io(&cs[3], NULL, &static_ops, &cs3val,

                          "sx1.cs3", OMAP_CS3_SIZE);

    memory_region_add_subregion(address_space,

                                OMAP_CS2_BASE, &cs[3]);



    fl_idx = 0;

#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif



    if ((dinfo = drive_get(IF_PFLASH, 0, fl_idx)) != NULL) {

        if (!pflash_cfi01_register(OMAP_CS0_BASE, NULL,

                                   "omap_sx1.flash0-1", flash_size,

                                   blk_by_legacy_dinfo(dinfo),

                                   sector_size, flash_size / sector_size,

                                   4, 0, 0, 0, 0, be)) {

            fprintf(stderr, "qemu: Error registering flash memory %d.\n",

                           fl_idx);

        }

        fl_idx++;

    }



    if ((version == 1) &&

            (dinfo = drive_get(IF_PFLASH, 0, fl_idx)) != NULL) {

        memory_region_init_ram(flash_1, NULL, "omap_sx1.flash1-0", flash1_size,

                               &error_abort);

        vmstate_register_ram_global(flash_1);

        memory_region_set_readonly(flash_1, true);

        memory_region_add_subregion(address_space, OMAP_CS1_BASE, flash_1);



        memory_region_init_io(&cs[1], NULL, &static_ops, &cs1val,

                              "sx1.cs1", OMAP_CS1_SIZE - flash1_size);

        memory_region_add_subregion(address_space,

                                OMAP_CS1_BASE + flash1_size, &cs[1]);



        if (!pflash_cfi01_register(OMAP_CS1_BASE, NULL,

                                   "omap_sx1.flash1-1", flash1_size,

                                   blk_by_legacy_dinfo(dinfo),

                                   sector_size, flash1_size / sector_size,

                                   4, 0, 0, 0, 0, be)) {

            fprintf(stderr, "qemu: Error registering flash memory %d.\n",

                           fl_idx);

        }

        fl_idx++;

    } else {

        memory_region_init_io(&cs[1], NULL, &static_ops, &cs1val,

                              "sx1.cs1", OMAP_CS1_SIZE);

        memory_region_add_subregion(address_space,

                                OMAP_CS1_BASE, &cs[1]);

    }



    if (!machine->kernel_filename && !fl_idx && !qtest_enabled()) {

        fprintf(stderr, "Kernel or Flash image must be specified\n");

        exit(1);

    }



    /* Load the kernel.  */

    sx1_binfo.kernel_filename = machine->kernel_filename;

    sx1_binfo.kernel_cmdline = machine->kernel_cmdline;

    sx1_binfo.initrd_filename = machine->initrd_filename;

    arm_load_kernel(mpu->cpu, &sx1_binfo);



    /* TODO: fix next line */

    //~ qemu_console_resize(ds, 640, 480);

}
