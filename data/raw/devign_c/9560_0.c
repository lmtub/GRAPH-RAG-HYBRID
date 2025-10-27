static void pc_init1(ram_addr_t ram_size,

                     const char *boot_device,

                     const char *kernel_filename,

                     const char *kernel_cmdline,

                     const char *initrd_filename,

                     const char *cpu_model,

                     int pci_enabled)

{

    char *filename;

    int ret, linux_boot, i;

    ram_addr_t ram_addr, bios_offset, option_rom_offset;

    ram_addr_t below_4g_mem_size, above_4g_mem_size = 0;

    int bios_size, isa_bios_size;

    PCIBus *pci_bus;

    ISADevice *isa_dev;

    int piix3_devfn = -1;

    CPUState *env;

    qemu_irq *cpu_irq;

    qemu_irq *isa_irq;

    qemu_irq *i8259;

    IsaIrqState *isa_irq_state;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DriveInfo *fd[MAX_FD];

    void *fw_cfg;



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



    for (i = 0; i < smp_cpus; i++) {

        env = pc_new_cpu(cpu_model);

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

    ret = rom_add_file_fixed(bios_name, (uint32_t)(-bios_size));

    if (ret != 0) {

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







    rom_enable_driver_roms = 1;

    option_rom_offset = qemu_ram_alloc(PC_ROM_SIZE);

    cpu_register_physical_memory(PC_ROM_MIN_VGA, PC_ROM_SIZE, option_rom_offset);



    /* map all the bios at the top of memory */

    cpu_register_physical_memory((uint32_t)(-bios_size),

                                 bios_size, bios_offset | IO_MEM_ROM);



    fw_cfg = bochs_bios_init();



    if (linux_boot) {

        load_linux(fw_cfg, kernel_filename, initrd_filename, kernel_cmdline, below_4g_mem_size);

    }



    for (i = 0; i < nb_option_roms; i++) {

        rom_add_option(option_rom[i]);

    }



    cpu_irq = qemu_allocate_irqs(pic_irq_request, NULL, 1);

    i8259 = i8259_init(cpu_irq[0]);

    isa_irq_state = qemu_mallocz(sizeof(*isa_irq_state));

    isa_irq_state->i8259 = i8259;

    isa_irq = qemu_allocate_irqs(isa_irq_handler, isa_irq_state, 24);



    if (pci_enabled) {

        pci_bus = i440fx_init(&i440fx_state, &piix3_devfn, isa_irq);

    } else {

        pci_bus = NULL;

        isa_bus_new(NULL);

    }

    isa_bus_irqs(isa_irq);



    ferr_irq = isa_reserve_irq(13);



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



    rtc_state = rtc_init(2000);



    qemu_register_boot_set(pc_boot_set, rtc_state);



    register_ioport_read(0x92, 1, 1, ioport92_read, NULL);

    register_ioport_write(0x92, 1, 1, ioport92_write, NULL);



    if (pci_enabled) {

        isa_irq_state->ioapic = ioapic_init();

    }

    pit = pit_init(0x40, isa_reserve_irq(0));

    pcspk_init(pit);

    if (!no_hpet) {

        hpet_init(isa_irq);

    }



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_isa_init(i, serial_hds[i]);

        }

    }



    for(i = 0; i < MAX_PARALLEL_PORTS; i++) {

        if (parallel_hds[i]) {

            parallel_init(i, parallel_hds[i]);

        }

    }



    for(i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];



        if (!pci_enabled || (nd->model && strcmp(nd->model, "ne2k_isa") == 0))

            pc_init_ne2k_isa(nd);

        else

            pci_nic_init_nofail(nd, "e1000", NULL);

    }



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }



    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        hd[i] = drive_get(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

    }



    if (pci_enabled) {

        pci_piix3_ide_init(pci_bus, hd, piix3_devfn + 1);

    } else {

        for(i = 0; i < MAX_IDE_BUS; i++) {

            isa_ide_init(ide_iobase[i], ide_iobase2[i], ide_irq[i],

	                 hd[MAX_IDE_DEVS * i], hd[MAX_IDE_DEVS * i + 1]);

        }

    }



    isa_dev = isa_create_simple("i8042");

    DMA_init(0);

#ifdef HAS_AUDIO

    audio_init(pci_enabled ? pci_bus : NULL, isa_irq);

#endif



    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    floppy_controller = fdctrl_init_isa(fd);



    cmos_init(below_4g_mem_size, above_4g_mem_size, boot_device, hd);



    if (pci_enabled && usb_enabled) {

        usb_uhci_piix3_init(pci_bus, piix3_devfn + 2);

    }



    if (pci_enabled && acpi_enabled) {

        uint8_t *eeprom_buf = qemu_mallocz(8 * 256); /* XXX: make this persistent */

        i2c_bus *smbus;



        /* TODO: Populate SPD eeprom data.  */

        smbus = piix4_pm_init(pci_bus, piix3_devfn + 3, 0xb100,

                              isa_reserve_irq(9));

        for (i = 0; i < 8; i++) {

            DeviceState *eeprom;

            eeprom = qdev_create((BusState *)smbus, "smbus-eeprom");

            qdev_prop_set_uint8(eeprom, "address", 0x50 + i);

            qdev_prop_set_ptr(eeprom, "data", eeprom_buf + (i * 256));

            qdev_init_nofail(eeprom);

        }

        piix4_acpi_system_hot_add_init(pci_bus);

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



    /* Add virtio console devices */

    if (pci_enabled) {

        for(i = 0; i < MAX_VIRTIO_CONSOLES; i++) {

            if (virtcon_hds[i]) {

                pci_create_simple(pci_bus, -1, "virtio-console-pci");

            }

        }

    }



    rom_load_fw(fw_cfg);

}
