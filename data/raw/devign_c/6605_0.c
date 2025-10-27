static void ppc_prep_init (ram_addr_t ram_size,

                           const char *boot_device,

                           const char *kernel_filename,

                           const char *kernel_cmdline,

                           const char *initrd_filename,

                           const char *cpu_model)

{

    CPUState *env = NULL;

    char *filename;

    nvram_t nvram;

    M48t59State *m48t59;

    int PPC_io_memory;

    int linux_boot, i, nb_nics1, bios_size;

    ram_addr_t ram_offset, bios_offset;

    uint32_t kernel_base, initrd_base;

    long kernel_size, initrd_size;

    PCIBus *pci_bus;

    qemu_irq *i8259;

    qemu_irq *cpu_exit_irq;

    int ppc_boot_device;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DriveInfo *fd[MAX_FD];



    sysctrl = qemu_mallocz(sizeof(sysctrl_t));



    linux_boot = (kernel_filename != NULL);



    /* init CPUs */

    if (cpu_model == NULL)

        cpu_model = "602";

    for (i = 0; i < smp_cpus; i++) {

        env = cpu_init(cpu_model);

        if (!env) {

            fprintf(stderr, "Unable to find PowerPC CPU definition\n");

            exit(1);

        }

        if (env->flags & POWERPC_FLAG_RTC_CLK) {

            /* POWER / PowerPC 601 RTC clock frequency is 7.8125 MHz */

            cpu_ppc_tb_init(env, 7812500UL);

        } else {

            /* Set time-base frequency to 100 Mhz */

            cpu_ppc_tb_init(env, 100UL * 1000UL * 1000UL);

        }

        qemu_register_reset((QEMUResetHandler*)&cpu_reset, env);

    }



    /* allocate RAM */

    ram_offset = qemu_ram_alloc(NULL, "ppc_prep.ram", ram_size);

    cpu_register_physical_memory(0, ram_size, ram_offset);



    /* allocate and load BIOS */

    bios_offset = qemu_ram_alloc(NULL, "ppc_prep.bios", BIOS_SIZE);

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    if (filename) {

        bios_size = get_image_size(filename);

    } else {

        bios_size = -1;

    }

    if (bios_size > 0 && bios_size <= BIOS_SIZE) {

        target_phys_addr_t bios_addr;

        bios_size = (bios_size + 0xfff) & ~0xfff;

        bios_addr = (uint32_t)(-bios_size);

        cpu_register_physical_memory(bios_addr, bios_size,

                                     bios_offset | IO_MEM_ROM);

        bios_size = load_image_targphys(filename, bios_addr, bios_size);

    }

    if (bios_size < 0 || bios_size > BIOS_SIZE) {

        hw_error("qemu: could not load PPC PREP bios '%s'\n", bios_name);

    }

    if (filename) {

        qemu_free(filename);

    }



    if (linux_boot) {

        kernel_base = KERNEL_LOAD_ADDR;

        /* now we can load the kernel */

        kernel_size = load_image_targphys(kernel_filename, kernel_base,

                                          ram_size - kernel_base);

        if (kernel_size < 0) {

            hw_error("qemu: could not load kernel '%s'\n", kernel_filename);

            exit(1);

        }

        /* load initrd */

        if (initrd_filename) {

            initrd_base = INITRD_LOAD_ADDR;

            initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                              ram_size - initrd_base);

            if (initrd_size < 0) {

                hw_error("qemu: could not load initial ram disk '%s'\n",

                          initrd_filename);

            }

        } else {

            initrd_base = 0;

            initrd_size = 0;

        }

        ppc_boot_device = 'm';

    } else {

        kernel_base = 0;

        kernel_size = 0;

        initrd_base = 0;

        initrd_size = 0;

        ppc_boot_device = '\0';

        /* For now, OHW cannot boot from the network. */

        for (i = 0; boot_device[i] != '\0'; i++) {

            if (boot_device[i] >= 'a' && boot_device[i] <= 'f') {

                ppc_boot_device = boot_device[i];

                break;

            }

        }

        if (ppc_boot_device == '\0') {

            fprintf(stderr, "No valid boot device for Mac99 machine\n");

            exit(1);

        }

    }



    isa_mem_base = 0xc0000000;

    if (PPC_INPUT(env) != PPC_FLAGS_INPUT_6xx) {

        hw_error("Only 6xx bus is supported on PREP machine\n");

    }

    i8259 = i8259_init(first_cpu->irq_inputs[PPC6xx_INPUT_INT]);

    pci_bus = pci_prep_init(i8259);

    /* Hmm, prep has no pci-isa bridge ??? */

    isa_bus_new(NULL);

    isa_bus_irqs(i8259);

    //    pci_bus = i440fx_init();

    /* Register 8 MB of ISA IO space (needed for non-contiguous map) */

    PPC_io_memory = cpu_register_io_memory(PPC_prep_io_read,

                                           PPC_prep_io_write, sysctrl,

                                           DEVICE_LITTLE_ENDIAN);

    cpu_register_physical_memory(0x80000000, 0x00800000, PPC_io_memory);



    /* init basic PC hardware */

    pci_vga_init(pci_bus);

    //    openpic = openpic_init(0x00000000, 0xF0000000, 1);

    //    pit = pit_init(0x40, i8259[0]);

    rtc_init(2000, NULL);



    if (serial_hds[0])

        serial_isa_init(0, serial_hds[0]);

    nb_nics1 = nb_nics;

    if (nb_nics1 > NE2000_NB_MAX)

        nb_nics1 = NE2000_NB_MAX;

    for(i = 0; i < nb_nics1; i++) {

        if (nd_table[i].model == NULL) {

	    nd_table[i].model = qemu_strdup("ne2k_isa");

        }

        if (strcmp(nd_table[i].model, "ne2k_isa") == 0) {

            isa_ne2000_init(ne2000_io[i], ne2000_irq[i], &nd_table[i]);

        } else {

            pci_nic_init_nofail(&nd_table[i], "ne2k_pci", NULL);

        }

    }



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }



    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        hd[i] = drive_get(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

    }



    for(i = 0; i < 1/*MAX_IDE_BUS*/; i++) {

        isa_ide_init(ide_iobase[i], ide_iobase2[i], ide_irq[i],

                     hd[2 * i],

		     hd[2 * i + 1]);

    }

    isa_create_simple("i8042");



    cpu_exit_irq = qemu_allocate_irqs(cpu_request_exit, NULL, 1);

    DMA_init(1, cpu_exit_irq);



    //    SB16_init();



    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    fdctrl_init_isa(fd);



    /* Register speaker port */

    register_ioport_read(0x61, 1, 1, speaker_ioport_read, NULL);

    register_ioport_write(0x61, 1, 1, speaker_ioport_write, NULL);

    /* Register fake IO ports for PREP */

    sysctrl->reset_irq = first_cpu->irq_inputs[PPC6xx_INPUT_HRESET];

    register_ioport_read(0x398, 2, 1, &PREP_io_read, sysctrl);

    register_ioport_write(0x398, 2, 1, &PREP_io_write, sysctrl);

    /* System control ports */

    register_ioport_read(0x0092, 0x01, 1, &PREP_io_800_readb, sysctrl);

    register_ioport_write(0x0092, 0x01, 1, &PREP_io_800_writeb, sysctrl);

    register_ioport_read(0x0800, 0x52, 1, &PREP_io_800_readb, sysctrl);

    register_ioport_write(0x0800, 0x52, 1, &PREP_io_800_writeb, sysctrl);

    /* PCI intack location */

    PPC_io_memory = cpu_register_io_memory(PPC_intack_read,

                                           PPC_intack_write, NULL,

                                           DEVICE_LITTLE_ENDIAN);

    cpu_register_physical_memory(0xBFFFFFF0, 0x4, PPC_io_memory);

    /* PowerPC control and status register group */

#if 0

    PPC_io_memory = cpu_register_io_memory(PPC_XCSR_read, PPC_XCSR_write,

                                           NULL, DEVICE_LITTLE_ENDIAN);

    cpu_register_physical_memory(0xFEFF0000, 0x1000, PPC_io_memory);

#endif



    if (usb_enabled) {

        usb_ohci_init_pci(pci_bus, -1);

    }



    m48t59 = m48t59_init(i8259[8], 0, 0x0074, NVRAM_SIZE, 59);

    if (m48t59 == NULL)

        return;

    sysctrl->nvram = m48t59;



    /* Initialise NVRAM */

    nvram.opaque = m48t59;

    nvram.read_fn = &m48t59_read;

    nvram.write_fn = &m48t59_write;

    PPC_NVRAM_set_params(&nvram, NVRAM_SIZE, "PREP", ram_size, ppc_boot_device,

                         kernel_base, kernel_size,

                         kernel_cmdline,

                         initrd_base, initrd_size,

                         /* XXX: need an option to load a NVRAM image */

                         0,

                         graphic_width, graphic_height, graphic_depth);



    /* Special port to get debug messages from Open-Firmware */

    register_ioport_write(0x0F00, 4, 1, &PPC_debug_write, NULL);

}
