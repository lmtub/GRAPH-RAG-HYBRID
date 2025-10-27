static void taihu_405ep_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *kernel_filename = machine->kernel_filename;

    const char *initrd_filename = machine->initrd_filename;

    char *filename;

    qemu_irq *pic;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *bios;

    MemoryRegion *ram_memories = g_malloc(2 * sizeof(*ram_memories));

    MemoryRegion *ram = g_malloc0(sizeof(*ram));

    hwaddr ram_bases[2], ram_sizes[2];

    long bios_size;

    target_ulong kernel_base, initrd_base;

    long kernel_size, initrd_size;

    int linux_boot;

    int fl_idx, fl_sectors;

    DriveInfo *dinfo;



    /* RAM is soldered to the board so the size cannot be changed */

    ram_size = 0x08000000;

    memory_region_allocate_system_memory(ram, NULL, "taihu_405ep.ram",

                                         ram_size);



    ram_bases[0] = 0;

    ram_sizes[0] = 0x04000000;

    memory_region_init_alias(&ram_memories[0], NULL,

                             "taihu_405ep.ram-0", ram, ram_bases[0],

                             ram_sizes[0]);

    ram_bases[1] = 0x04000000;

    ram_sizes[1] = 0x04000000;

    memory_region_init_alias(&ram_memories[1], NULL,

                             "taihu_405ep.ram-1", ram, ram_bases[1],

                             ram_sizes[1]);

#ifdef DEBUG_BOARD_INIT

    printf("%s: register cpu\n", __func__);

#endif

    ppc405ep_init(sysmem, ram_memories, ram_bases, ram_sizes,

                  33333333, &pic, kernel_filename == NULL ? 0 : 1);

    /* allocate and load BIOS */

#ifdef DEBUG_BOARD_INIT

    printf("%s: register BIOS\n", __func__);

#endif

    fl_idx = 0;

#if defined(USE_FLASH_BIOS)

    dinfo = drive_get(IF_PFLASH, 0, fl_idx);

    if (dinfo) {

        BlockDriverState *bs = blk_bs(blk_by_legacy_dinfo(dinfo));



        bios_size = bdrv_getlength(bs);

        /* XXX: should check that size is 2MB */

        //        bios_size = 2 * 1024 * 1024;

        fl_sectors = (bios_size + 65535) >> 16;

#ifdef DEBUG_BOARD_INIT

        printf("Register parallel flash %d size %lx"

               " at addr %lx '%s' %d\n",

               fl_idx, bios_size, -bios_size,

               bdrv_get_device_name(bs), fl_sectors);

#endif

        pflash_cfi02_register((uint32_t)(-bios_size),

                              NULL, "taihu_405ep.bios", bios_size,

                              bs, 65536, fl_sectors, 1,

                              4, 0x0001, 0x22DA, 0x0000, 0x0000, 0x555, 0x2AA,

                              1);

        fl_idx++;

    } else

#endif

    {

#ifdef DEBUG_BOARD_INIT

        printf("Load BIOS from file\n");

#endif

        if (bios_name == NULL)

            bios_name = BIOS_FILENAME;

        bios = g_new(MemoryRegion, 1);

        memory_region_init_ram(bios, NULL, "taihu_405ep.bios", BIOS_SIZE,

                               &error_abort);

        vmstate_register_ram_global(bios);

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (filename) {

            bios_size = load_image(filename, memory_region_get_ram_ptr(bios));

            g_free(filename);

            if (bios_size < 0 || bios_size > BIOS_SIZE) {

                error_report("Could not load PowerPC BIOS '%s'", bios_name);

                exit(1);

            }

            bios_size = (bios_size + 0xfff) & ~0xfff;

            memory_region_add_subregion(sysmem, (uint32_t)(-bios_size), bios);

        } else if (!qtest_enabled()) {

            error_report("Could not load PowerPC BIOS '%s'", bios_name);

            exit(1);

        }

        memory_region_set_readonly(bios, true);

    }

    /* Register Linux flash */

    dinfo = drive_get(IF_PFLASH, 0, fl_idx);

    if (dinfo) {

        BlockDriverState *bs = blk_bs(blk_by_legacy_dinfo(dinfo));



        bios_size = bdrv_getlength(bs);

        /* XXX: should check that size is 32MB */

        bios_size = 32 * 1024 * 1024;

        fl_sectors = (bios_size + 65535) >> 16;

#ifdef DEBUG_BOARD_INIT

        printf("Register parallel flash %d size %lx"

               " at addr " TARGET_FMT_lx " '%s'\n",

               fl_idx, bios_size, (target_ulong)0xfc000000,

               bdrv_get_device_name(bs));

#endif

        pflash_cfi02_register(0xfc000000, NULL, "taihu_405ep.flash", bios_size,

                              bs, 65536, fl_sectors, 1,

                              4, 0x0001, 0x22DA, 0x0000, 0x0000, 0x555, 0x2AA,

                              1);

        fl_idx++;

    }

    /* Register CLPD & LCD display */

#ifdef DEBUG_BOARD_INIT

    printf("%s: register CPLD\n", __func__);

#endif

    taihu_cpld_init(sysmem, 0x50100000);

    /* Load kernel */

    linux_boot = (kernel_filename != NULL);

    if (linux_boot) {

#ifdef DEBUG_BOARD_INIT

        printf("%s: load kernel\n", __func__);

#endif

        kernel_base = KERNEL_LOAD_ADDR;

        /* now we can load the kernel */

        kernel_size = load_image_targphys(kernel_filename, kernel_base,

                                          ram_size - kernel_base);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

        /* load initrd */

        if (initrd_filename) {

            initrd_base = INITRD_LOAD_ADDR;

            initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                              ram_size - initrd_base);

            if (initrd_size < 0) {

                fprintf(stderr,

                        "qemu: could not load initial ram disk '%s'\n",

                        initrd_filename);

                exit(1);

            }

        } else {

            initrd_base = 0;

            initrd_size = 0;

        }

    } else {

        kernel_base = 0;

        kernel_size = 0;

        initrd_base = 0;

        initrd_size = 0;

    }

#ifdef DEBUG_BOARD_INIT

    printf("%s: Done\n", __func__);

#endif

}
