void mips_r4k_init (ram_addr_t ram_size,

                    const char *boot_device,

                    const char *kernel_filename, const char *kernel_cmdline,

                    const char *initrd_filename, const char *cpu_model)

{

    char *filename;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *bios;

    MemoryRegion *iomem = g_new(MemoryRegion, 1);

    int bios_size;

    CPUState *env;

    ResetData *reset_info;

    int i;

    qemu_irq *i8259;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DriveInfo *dinfo;

    int be;



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_MIPS64

        cpu_model = "R4000";

#else

        cpu_model = "24Kf";

#endif

    }

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    reset_info = g_malloc0(sizeof(ResetData));

    reset_info->env = env;

    reset_info->vector = env->active_tc.PC;

    qemu_register_reset(main_cpu_reset, reset_info);



    /* allocate RAM */

    if (ram_size > (256 << 20)) {

        fprintf(stderr,

                "qemu: Too much memory for this machine: %d MB, maximum 256 MB\n",

                ((unsigned int)ram_size / (1 << 20)));

        exit(1);

    }

    memory_region_init_ram(ram, NULL, "mips_r4k.ram", ram_size);



    memory_region_add_subregion(address_space_mem, 0, ram);



    memory_region_init_io(iomem, &mips_qemu_ops, NULL, "mips-qemu", 0x10000);

    memory_region_add_subregion(address_space_mem, 0x1fbf0000, iomem);



    /* Try to load a BIOS image. If this fails, we continue regardless,

       but initialize the hardware ourselves. When a kernel gets

       preloaded we also initialize the hardware, since the BIOS wasn't

       run. */

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    if (filename) {

        bios_size = get_image_size(filename);

    } else {

        bios_size = -1;

    }

#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    if ((bios_size > 0) && (bios_size <= BIOS_SIZE)) {

        bios = g_new(MemoryRegion, 1);

        memory_region_init_ram(bios, NULL, "mips_r4k.bios", BIOS_SIZE);

        memory_region_set_readonly(bios, true);

        memory_region_add_subregion(get_system_memory(), 0x1fc00000, bios);



        load_image_targphys(filename, 0x1fc00000, BIOS_SIZE);

    } else if ((dinfo = drive_get(IF_PFLASH, 0, 0)) != NULL) {

        uint32_t mips_rom = 0x00400000;

        if (!pflash_cfi01_register(0x1fc00000, NULL, "mips_r4k.bios", mips_rom,

                                   dinfo->bdrv, sector_len,

                                   mips_rom / sector_len,

                                   4, 0, 0, 0, 0, be)) {

            fprintf(stderr, "qemu: Error registering flash memory.\n");

	}

    }

    else {

	/* not fatal */

        fprintf(stderr, "qemu: Warning, could not load MIPS bios '%s'\n",

		bios_name);

    }

    if (filename) {

        g_free(filename);

    }



    if (kernel_filename) {

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        reset_info->vector = load_kernel();

    }



    /* Init CPU internal devices */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /* The PIC is attached to the MIPS CPU INT0 pin */

    i8259 = i8259_init(env->irq[2]);

    isa_bus_new(NULL, get_system_io());

    isa_bus_irqs(i8259);



    rtc_init(2000, NULL);



    /* Register 64 KB of ISA IO space at 0x14000000 */

    isa_mmio_init(0x14000000, 0x00010000);

    isa_mem_base = 0x10000000;



    pit = pit_init(0x40, 0);



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_isa_init(i, serial_hds[i]);

        }

    }



    isa_vga_init();



    if (nd_table[0].vlan)

        isa_ne2000_init(0x300, 9, &nd_table[0]);



    ide_drive_get(hd, MAX_IDE_BUS);

    for(i = 0; i < MAX_IDE_BUS; i++)

        isa_ide_init(ide_iobase[i], ide_iobase2[i], ide_irq[i],

                     hd[MAX_IDE_DEVS * i],

		     hd[MAX_IDE_DEVS * i + 1]);



    isa_create_simple("i8042");

}
