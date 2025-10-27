void mips_malta_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    char *filename;

    pflash_t *fl;

    MemoryRegion *system_memory = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *bios, *bios_copy = g_new(MemoryRegion, 1);

    target_long bios_size = FLASH_SIZE;

    int64_t kernel_entry;

    PCIBus *pci_bus;

    ISABus *isa_bus;

    MIPSCPU *cpu;

    CPUMIPSState *env;

    qemu_irq *isa_irq;

    qemu_irq *cpu_exit_irq;

    int piix4_devfn;

    i2c_bus *smbus;

    int i;

    DriveInfo *dinfo;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DriveInfo *fd[MAX_FD];

    int fl_idx = 0;

    int fl_sectors = bios_size >> 16;

    int be;



    DeviceState *dev = qdev_create(NULL, TYPE_MIPS_MALTA);

    MaltaState *s = MIPS_MALTA(dev);



    qdev_init_nofail(dev);



    /* Make sure the first 3 serial ports are associated with a device. */

    for(i = 0; i < 3; i++) {

        if (!serial_hds[i]) {

            char label[32];

            snprintf(label, sizeof(label), "serial%d", i);

            serial_hds[i] = qemu_chr_new(label, "null", NULL);

        }

    }



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_MIPS64

        cpu_model = "20Kc";

#else

        cpu_model = "24Kf";

#endif

    }



    for (i = 0; i < smp_cpus; i++) {

        cpu = cpu_mips_init(cpu_model);

        if (cpu == NULL) {

            fprintf(stderr, "Unable to find CPU definition\n");

            exit(1);

        }

        env = &cpu->env;



        /* Init internal devices */

        cpu_mips_irq_init_cpu(env);

        cpu_mips_clock_init(env);

        qemu_register_reset(main_cpu_reset, cpu);

    }

    cpu = MIPS_CPU(first_cpu);

    env = &cpu->env;



    /* allocate RAM */

    if (ram_size > (256 << 20)) {

        fprintf(stderr,

                "qemu: Too much memory for this machine: %d MB, maximum 256 MB\n",

                ((unsigned int)ram_size / (1 << 20)));

        exit(1);

    }

    memory_region_init_ram(ram, NULL, "mips_malta.ram", ram_size);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(system_memory, 0, ram);



    /* generate SPD EEPROM data */

    eeprom_generate(&eeprom, ram_size);



#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    /* FPGA */

    /* The CBUS UART is attached to the MIPS CPU INT2 pin, ie interrupt 4 */

    malta_fpga_init(system_memory, FPGA_ADDRESS, env->irq[4], serial_hds[2]);



    /* Load firmware in flash / BIOS. */

    dinfo = drive_get(IF_PFLASH, 0, fl_idx);

#ifdef DEBUG_BOARD_INIT

    if (dinfo) {

        printf("Register parallel flash %d size " TARGET_FMT_lx " at "

               "addr %08llx '%s' %x\n",

               fl_idx, bios_size, FLASH_ADDRESS,

               bdrv_get_device_name(dinfo->bdrv), fl_sectors);

    }

#endif

    fl = pflash_cfi01_register(FLASH_ADDRESS, NULL, "mips_malta.bios",

                               BIOS_SIZE, dinfo ? dinfo->bdrv : NULL,

                               65536, fl_sectors,

                               4, 0x0000, 0x0000, 0x0000, 0x0000, be);

    bios = pflash_cfi01_get_memory(fl);

    fl_idx++;

    if (kernel_filename) {

        /* Write a small bootloader to the flash location. */

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        kernel_entry = load_kernel();

        write_bootloader(env, memory_region_get_ram_ptr(bios), kernel_entry);

    } else {

        /* Load firmware from flash. */

        if (!dinfo) {

            /* Load a BIOS image. */

            if (bios_name == NULL) {

                bios_name = BIOS_FILENAME;

            }

            filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

            if (filename) {

                bios_size = load_image_targphys(filename, FLASH_ADDRESS,

                                                BIOS_SIZE);

                g_free(filename);

            } else {

                bios_size = -1;

            }

            if ((bios_size < 0 || bios_size > BIOS_SIZE) && !kernel_filename) {

                fprintf(stderr,

                        "qemu: Warning, could not load MIPS bios '%s', and no -kernel argument was specified\n",

                        bios_name);

            }

        }

        /* In little endian mode the 32bit words in the bios are swapped,

           a neat trick which allows bi-endian firmware. */

#ifndef TARGET_WORDS_BIGENDIAN

        {

            uint32_t *end, *addr = rom_ptr(FLASH_ADDRESS);

            if (!addr) {

                addr = memory_region_get_ram_ptr(bios);

            }

            end = (void *)addr + MIN(bios_size, 0x3e0000);

            while (addr < end) {

                bswap32s(addr);

                addr++;

            }

        }

#endif

    }



    /*

     * Map the BIOS at a 2nd physical location, as on the real board.

     * Copy it so that we can patch in the MIPS revision, which cannot be

     * handled by an overlapping region as the resulting ROM code subpage

     * regions are not executable.

     */

    memory_region_init_ram(bios_copy, NULL, "bios.1fc", BIOS_SIZE);

    if (!rom_copy(memory_region_get_ram_ptr(bios_copy),

                  FLASH_ADDRESS, bios_size)) {

        memcpy(memory_region_get_ram_ptr(bios_copy),

               memory_region_get_ram_ptr(bios), bios_size);

    }

    memory_region_set_readonly(bios_copy, true);

    memory_region_add_subregion(system_memory, RESET_ADDRESS, bios_copy);



    /* Board ID = 0x420 (Malta Board with CoreLV) */

    stl_p(memory_region_get_ram_ptr(bios_copy) + 0x10, 0x00000420);



    /* Init internal devices */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /*

     * We have a circular dependency problem: pci_bus depends on isa_irq,

     * isa_irq is provided by i8259, i8259 depends on ISA, ISA depends

     * on piix4, and piix4 depends on pci_bus.  To stop the cycle we have

     * qemu_irq_proxy() adds an extra bit of indirection, allowing us

     * to resolve the isa_irq -> i8259 dependency after i8259 is initialized.

     */

    isa_irq = qemu_irq_proxy(&s->i8259, 16);



    /* Northbridge */

    pci_bus = gt64120_register(isa_irq);



    /* Southbridge */

    ide_drive_get(hd, MAX_IDE_BUS);



    piix4_devfn = piix4_init(pci_bus, &isa_bus, 80);



    /* Interrupt controller */

    /* The 8259 is attached to the MIPS CPU INT0 pin, ie interrupt 2 */

    s->i8259 = i8259_init(isa_bus, env->irq[2]);



    isa_bus_irqs(isa_bus, s->i8259);

    pci_piix4_ide_init(pci_bus, hd, piix4_devfn + 1);

    pci_create_simple(pci_bus, piix4_devfn + 2, "piix4-usb-uhci");

    smbus = piix4_pm_init(pci_bus, piix4_devfn + 3, 0x1100,

                          isa_get_irq(NULL, 9), NULL, 0, NULL);

    /* TODO: Populate SPD eeprom data.  */

    smbus_eeprom_init(smbus, 8, NULL, 0);

    pit = pit_init(isa_bus, 0x40, 0, NULL);

    cpu_exit_irq = qemu_allocate_irqs(cpu_request_exit, NULL, 1);

    DMA_init(0, cpu_exit_irq);



    /* Super I/O */

    isa_create_simple(isa_bus, "i8042");



    rtc_init(isa_bus, 2000, NULL);

    serial_isa_init(isa_bus, 0, serial_hds[0]);

    serial_isa_init(isa_bus, 1, serial_hds[1]);

    if (parallel_hds[0])

        parallel_init(isa_bus, 0, parallel_hds[0]);

    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    fdctrl_init_isa(isa_bus, fd);



    /* Network card */

    network_init(pci_bus);



    /* Optional PCI video card */

    pci_vga_init(pci_bus);

}
