static void pc_init1(ram_addr_t ram_size,

                     const char *boot_device,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename,

                     int pci_enabled, const char *cpu_model)

{

    char *filename;

    int ret, linux_boot, i;

    ram_addr_t ram_addr, bios_offset, option_rom_offset;

    ram_addr_t below_4g_mem_size, above_4g_mem_size = 0;

    int bios_size, isa_bios_size, oprom_area_size;

    PCIBus *pci_bus;

    int piix3_devfn = -1;

    CPUState *env;

    qemu_irq *cpu_irq;

    qemu_irq *i8259;

    int index;

    BlockDriverState *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    BlockDriverState *fd[MAX_FD];

    int using_vga = cirrus_vga_enabled || std_vga_enabled || vmsvga_enabled;



    if (ram_size >= 0xe0000000 ) {

        above_4g_mem_size = ram_size - 0xe0000000;

        below_4g_mem_size = 0xe0000000;

    } else {

        below_4g_mem_size = ram_size;

    }



    linux_boot = (kernel_filename != NULL);



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_X86_64

        cpu_model = "qemu64";

#else

        cpu_model = "qemu32";

#endif

    }

    

    for(i = 0; i < smp_cpus; i++) {

        env = cpu_init(cpu_model);

        if (!env) {

            fprintf(stderr, "Unable to find x86 CPU definition\n");

            exit(1);

        }

        if ((env->cpuid_features & CPUID_APIC) || smp_cpus > 1) {

            env->cpuid_apic_id = env->cpu_index;

            apic_init(env);

        }

        qemu_register_reset(main_cpu_reset, 0, env);

    }



    vmport_init();



    /* allocate RAM */

    ram_addr = qemu_ram_alloc(0xa0000);

    cpu_register_physical_memory(0, 0xa0000, ram_addr);



    /* Allocate, even though we won't register, so we don't break the

     * phys_ram_base + PA assumption. This range includes vga (0xa0000 - 0xc0000),

     * and some bios areas, which will be registered later

     */

    ram_addr = qemu_ram_alloc(0x100000 - 0xa0000);

    ram_addr = qemu_ram_alloc(below_4g_mem_size - 0x100000);

    cpu_register_physical_memory(0x100000,

                 below_4g_mem_size - 0x100000,

                 ram_addr);



    /* above 4giga memory allocation */

    if (above_4g_mem_size > 0) {

#if TARGET_PHYS_ADDR_BITS == 32

        hw_error("To much RAM for 32-bit physical address");

#else

        ram_addr = qemu_ram_alloc(above_4g_mem_size);

        cpu_register_physical_memory(0x100000000ULL,

                                     above_4g_mem_size,

                                     ram_addr);

#endif

    }





    /* BIOS load */

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    if (filename) {

        bios_size = get_image_size(filename);

    } else {

        bios_size = -1;

    }

    if (bios_size <= 0 ||

        (bios_size % 65536) != 0) {

        goto bios_error;

    }

    bios_offset = qemu_ram_alloc(bios_size);

    ret = load_image(filename, qemu_get_ram_ptr(bios_offset));

    if (ret != bios_size) {

    bios_error:

        fprintf(stderr, "qemu: could not load PC BIOS '%s'\n", bios_name);

        exit(1);

    }

    if (filename) {

        qemu_free(filename);

    }

    /* map the last 128KB of the BIOS in ISA space */

    isa_bios_size = bios_size;

    if (isa_bios_size > (128 * 1024))

        isa_bios_size = 128 * 1024;

    cpu_register_physical_memory(0x100000 - isa_bios_size,

                                 isa_bios_size,

                                 (bios_offset + bios_size - isa_bios_size) | IO_MEM_ROM);







    option_rom_offset = qemu_ram_alloc(0x20000);

    oprom_area_size = 0;

    cpu_register_physical_memory(0xc0000, 0x20000, option_rom_offset);



    if (using_vga) {

        const char *vgabios_filename;

        /* VGA BIOS load */

        if (cirrus_vga_enabled) {

            vgabios_filename = VGABIOS_CIRRUS_FILENAME;

        } else {

            vgabios_filename = VGABIOS_FILENAME;

        }

        oprom_area_size = load_option_rom(vgabios_filename, 0xc0000, 0xe0000);

    }

    /* Although video roms can grow larger than 0x8000, the area between

     * 0xc0000 - 0xc8000 is reserved for them. It means we won't be looking

     * for any other kind of option rom inside this area */

    if (oprom_area_size < 0x8000)

        oprom_area_size = 0x8000;



    if (linux_boot) {

        load_linux(0xc0000 + oprom_area_size,

                   kernel_filename, initrd_filename, kernel_cmdline, below_4g_mem_size);

        oprom_area_size += 2048;

    }



    for (i = 0; i < nb_option_roms; i++) {

        oprom_area_size += load_option_rom(option_rom[i],

                                           0xc0000 + oprom_area_size, 0xe0000);

    }



    /* map all the bios at the top of memory */

    cpu_register_physical_memory((uint32_t)(-bios_size),

                                 bios_size, bios_offset | IO_MEM_ROM);



    bochs_bios_init();



    cpu_irq = qemu_allocate_irqs(pic_irq_request, NULL, 1);

    i8259 = i8259_init(cpu_irq[0]);

    ferr_irq = i8259[13];



    if (pci_enabled) {

        pci_bus = i440fx_init(&i440fx_state, i8259);

        piix3_devfn = piix3_init(pci_bus, -1);

    } else {

        pci_bus = NULL;

    }



    /* init basic PC hardware */

    register_ioport_write(0x80, 1, 1, ioport80_write, NULL);



    register_ioport_write(0xf0, 1, 1, ioportF0_write, NULL);



    if (cirrus_vga_enabled) {

        if (pci_enabled) {

            pci_cirrus_vga_init(pci_bus);

        } else {

            isa_cirrus_vga_init();

        }

    } else if (vmsvga_enabled) {

        if (pci_enabled)

            pci_vmsvga_init(pci_bus);

        else

            fprintf(stderr, "%s: vmware_vga: no PCI bus\n", __FUNCTION__);

    } else if (std_vga_enabled) {

        if (pci_enabled) {

            pci_vga_init(pci_bus, 0, 0);

        } else {

            isa_vga_init();

        }

    }



    rtc_state = rtc_init(0x70, i8259[8], 2000);



    qemu_register_boot_set(pc_boot_set, rtc_state);



    register_ioport_read(0x92, 1, 1, ioport92_read, NULL);

    register_ioport_write(0x92, 1, 1, ioport92_write, NULL);



    if (pci_enabled) {

        ioapic = ioapic_init();

    }

    pit = pit_init(0x40, i8259[0]);

    pcspk_init(pit);

    if (!no_hpet) {

        hpet_init(i8259);

    }

    if (pci_enabled) {

        pic_set_alt_irq_func(isa_pic, ioapic_set_irq, ioapic);

    }



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_init(serial_io[i], i8259[serial_irq[i]], 115200,

                        serial_hds[i]);

        }

    }



    for(i = 0; i < MAX_PARALLEL_PORTS; i++) {

        if (parallel_hds[i]) {

            parallel_init(parallel_io[i], i8259[parallel_irq[i]],

                          parallel_hds[i]);

        }

    }



    watchdog_pc_init(pci_bus);



    for(i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];



        if (!pci_enabled || (nd->model && strcmp(nd->model, "ne2k_isa") == 0))

            pc_init_ne2k_isa(nd, i8259);

        else

            pci_nic_init(pci_bus, nd, -1, "ne2k_pci");

    }



    qemu_system_hot_add_init();



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }



    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        index = drive_get_index(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

	if (index != -1)

	    hd[i] = drives_table[index].bdrv;

	else

	    hd[i] = NULL;

    }



    if (pci_enabled) {

        pci_piix3_ide_init(pci_bus, hd, piix3_devfn + 1, i8259);

    } else {

        for(i = 0; i < MAX_IDE_BUS; i++) {

            isa_ide_init(ide_iobase[i], ide_iobase2[i], i8259[ide_irq[i]],

	                 hd[MAX_IDE_DEVS * i], hd[MAX_IDE_DEVS * i + 1]);

        }

    }



    i8042_init(i8259[1], i8259[12], 0x60);

    DMA_init(0);

#ifdef HAS_AUDIO

    audio_init(pci_enabled ? pci_bus : NULL, i8259);

#endif



    for(i = 0; i < MAX_FD; i++) {

        index = drive_get_index(IF_FLOPPY, 0, i);

	if (index != -1)

	    fd[i] = drives_table[index].bdrv;

	else

	    fd[i] = NULL;

    }

    floppy_controller = fdctrl_init(i8259[6], 2, 0, 0x3f0, fd);



    cmos_init(below_4g_mem_size, above_4g_mem_size, boot_device, hd);



    if (pci_enabled && usb_enabled) {

        usb_uhci_piix3_init(pci_bus, piix3_devfn + 2);

    }



    if (pci_enabled && acpi_enabled) {

        uint8_t *eeprom_buf = qemu_mallocz(8 * 256); /* XXX: make this persistent */

        i2c_bus *smbus;



        /* TODO: Populate SPD eeprom data.  */

        smbus = piix4_pm_init(pci_bus, piix3_devfn + 3, 0xb100, i8259[9]);

        for (i = 0; i < 8; i++) {

            DeviceState *eeprom;

            eeprom = qdev_create((BusState *)smbus, "smbus-eeprom");

            qdev_set_prop_int(eeprom, "address", 0x50 + i);

            qdev_set_prop_ptr(eeprom, "data", eeprom_buf + (i * 256));

            qdev_init(eeprom);

        }

    }



    if (i440fx_state) {

        i440fx_init_memory_mappings(i440fx_state);

    }



    if (pci_enabled) {

	int max_bus;

        int bus;



        max_bus = drive_get_max_bus(IF_SCSI);

	for (bus = 0; bus <= max_bus; bus++) {

            pci_create_simple(pci_bus, -1, "lsi53c895a");

        }

    }



    /* Add virtio block devices */

    if (pci_enabled) {

        int index;

        int unit_id = 0;



        while ((index = drive_get_index(IF_VIRTIO, 0, unit_id)) != -1) {

            pci_create_simple(pci_bus, -1, "virtio-blk-pci");

            unit_id++;

        }

    }



    /* Add virtio balloon device */

    if (pci_enabled && !no_virtio_balloon) {

        pci_create_simple(pci_bus, -1, "virtio-balloon-pci");

    }



    /* Add virtio console devices */

    if (pci_enabled) {

        for(i = 0; i < MAX_VIRTIO_CONSOLES; i++) {

            if (virtcon_hds[i]) {

                pci_create_simple(pci_bus, -1, "virtio-console-pci");

            }

        }

    }

}
