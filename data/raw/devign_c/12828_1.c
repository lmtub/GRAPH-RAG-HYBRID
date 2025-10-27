static void pc_init1(ram_addr_t ram_size, int vga_ram_size,

                     const char *boot_device, DisplayState *ds,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename,

                     int pci_enabled, const char *cpu_model)

{

    char buf[1024];

    int ret, linux_boot, i;

    ram_addr_t ram_addr, vga_ram_addr, bios_offset, vga_bios_offset;

    ram_addr_t below_4g_mem_size, above_4g_mem_size = 0;

    int bios_size, isa_bios_size, vga_bios_size;

    PCIBus *pci_bus;

    int piix3_devfn = -1;

    CPUState *env;

    qemu_irq *cpu_irq;

    qemu_irq *i8259;

    int index;

    BlockDriverState *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    BlockDriverState *fd[MAX_FD];



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

        if (i != 0)

            env->halted = 1;

        if (smp_cpus > 1) {

            /* XXX: enable it in all cases */

            env->cpuid_features |= CPUID_APIC;

        }

        qemu_register_reset(main_cpu_reset, env);

        if (pci_enabled) {

            apic_init(env);

        }

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

        ram_addr = qemu_ram_alloc(above_4g_mem_size);

        cpu_register_physical_memory(0x100000000ULL,

                                     above_4g_mem_size,

                                     ram_addr);

    }





    /* allocate VGA RAM */

    vga_ram_addr = qemu_ram_alloc(vga_ram_size);



    /* BIOS load */

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    snprintf(buf, sizeof(buf), "%s/%s", bios_dir, bios_name);

    bios_size = get_image_size(buf);

    if (bios_size <= 0 ||

        (bios_size % 65536) != 0) {

        goto bios_error;

    }

    bios_offset = qemu_ram_alloc(bios_size);

    ret = load_image(buf, phys_ram_base + bios_offset);

    if (ret != bios_size) {

    bios_error:

        fprintf(stderr, "qemu: could not load PC BIOS '%s'\n", buf);

        exit(1);

    }



    if (cirrus_vga_enabled || std_vga_enabled || vmsvga_enabled) {

        /* VGA BIOS load */

        if (cirrus_vga_enabled) {

            snprintf(buf, sizeof(buf), "%s/%s", bios_dir, VGABIOS_CIRRUS_FILENAME);

        } else {

            snprintf(buf, sizeof(buf), "%s/%s", bios_dir, VGABIOS_FILENAME);

        }

        vga_bios_size = get_image_size(buf);

        if (vga_bios_size <= 0 || vga_bios_size > 65536)

            goto vga_bios_error;

        vga_bios_offset = qemu_ram_alloc(65536);



        ret = load_image(buf, phys_ram_base + vga_bios_offset);

        if (ret != vga_bios_size) {

vga_bios_error:

            fprintf(stderr, "qemu: could not load VGA BIOS '%s'\n", buf);

            exit(1);

        }

    }



    /* setup basic memory access */

    cpu_register_physical_memory(0xc0000, 0x10000,

                                 vga_bios_offset | IO_MEM_ROM);



    /* map the last 128KB of the BIOS in ISA space */

    isa_bios_size = bios_size;

    if (isa_bios_size > (128 * 1024))

        isa_bios_size = 128 * 1024;

    cpu_register_physical_memory(0x100000 - isa_bios_size,

                                 isa_bios_size,

                                 (bios_offset + bios_size - isa_bios_size) | IO_MEM_ROM);



    {

        ram_addr_t option_rom_offset;

        int size, offset;



        offset = 0;

        if (linux_boot) {

            option_rom_offset = qemu_ram_alloc(TARGET_PAGE_SIZE);

            load_linux(phys_ram_base + option_rom_offset,

                       kernel_filename, initrd_filename, kernel_cmdline);

            cpu_register_physical_memory(0xd0000, TARGET_PAGE_SIZE,

                                         option_rom_offset | IO_MEM_ROM);

            offset = TARGET_PAGE_SIZE;

        }



        for (i = 0; i < nb_option_roms; i++) {

            size = get_image_size(option_rom[i]);

            if (size < 0) {

                fprintf(stderr, "Could not load option rom '%s'\n",

                        option_rom[i]);

                exit(1);

            }

            if (size > (0x10000 - offset))

                goto option_rom_error;

            option_rom_offset = qemu_ram_alloc(size);

            ret = load_image(option_rom[i], phys_ram_base + option_rom_offset);

            if (ret != size) {

            option_rom_error:

                fprintf(stderr, "Too many option ROMS\n");

                exit(1);

            }

            size = (size + 4095) & ~4095;

            cpu_register_physical_memory(0xd0000 + offset,

                                         size, option_rom_offset | IO_MEM_ROM);

            offset += size;

        }

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

            pci_cirrus_vga_init(pci_bus,

                                ds, phys_ram_base + vga_ram_addr,

                                vga_ram_addr, vga_ram_size);

        } else {

            isa_cirrus_vga_init(ds, phys_ram_base + vga_ram_addr,

                                vga_ram_addr, vga_ram_size);

        }

    } else if (vmsvga_enabled) {

        if (pci_enabled)

            pci_vmsvga_init(pci_bus, ds, phys_ram_base + vga_ram_addr,

                            vga_ram_addr, vga_ram_size);

        else

            fprintf(stderr, "%s: vmware_vga: no PCI bus\n", __FUNCTION__);

    } else if (std_vga_enabled) {

        if (pci_enabled) {

            pci_vga_init(pci_bus, ds, phys_ram_base + vga_ram_addr,

                         vga_ram_addr, vga_ram_size, 0, 0);

        } else {

            isa_vga_init(ds, phys_ram_base + vga_ram_addr,

                         vga_ram_addr, vga_ram_size);

        }

    }



    rtc_state = rtc_init(0x70, i8259[8]);



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



    for(i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];



        if (!pci_enabled || (nd->model && strcmp(nd->model, "ne2k_isa") == 0))

            pc_init_ne2k_isa(nd, i8259);

        else

            pci_nic_init(pci_bus, nd, -1, "ne2k_pci");

    }



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

            smbus_eeprom_device_init(smbus, 0x50 + i, eeprom_buf + (i * 256));

        }

    }



    if (i440fx_state) {

        i440fx_init_memory_mappings(i440fx_state);

    }



    if (pci_enabled) {

	int max_bus;

        int bus, unit;

        void *scsi;



        max_bus = drive_get_max_bus(IF_SCSI);



	for (bus = 0; bus <= max_bus; bus++) {

            scsi = lsi_scsi_init(pci_bus, -1);

            for (unit = 0; unit < LSI_MAX_DEVS; unit++) {

	        index = drive_get_index(IF_SCSI, bus, unit);

		if (index == -1)

		    continue;

		lsi_scsi_attach(scsi, drives_table[index].bdrv, unit);

	    }

        }

    }



    /* Add virtio block devices */

    if (pci_enabled) {

        int index;

        int unit_id = 0;



        while ((index = drive_get_index(IF_VIRTIO, 0, unit_id)) != -1) {

            virtio_blk_init(pci_bus, drives_table[index].bdrv);

            unit_id++;

        }

    }



    /* Add virtio balloon device */

    if (pci_enabled)

        virtio_balloon_init(pci_bus);



    /* Add virtio console devices */

    if (pci_enabled) {

        for(i = 0; i < MAX_VIRTIO_CONSOLES; i++) {

            if (virtcon_hds[i])

                virtio_console_init(pci_bus, virtcon_hds[i]);

        }

    }

}
