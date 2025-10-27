static void ppc_prep_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    const char *boot_device = args->boot_order;

    MemoryRegion *sysmem = get_system_memory();

    PowerPCCPU *cpu = NULL;

    CPUPPCState *env = NULL;

    nvram_t nvram;

    M48t59State *m48t59;

    PortioList *port_list = g_new(PortioList, 1);

#if 0

    MemoryRegion *xcsr = g_new(MemoryRegion, 1);

#endif

    int linux_boot, i, nb_nics1;

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *vga = g_new(MemoryRegion, 1);

    uint32_t kernel_base, initrd_base;

    long kernel_size, initrd_size;

    DeviceState *dev;

    PCIHostState *pcihost;

    PCIBus *pci_bus;

    PCIDevice *pci;

    ISABus *isa_bus;

    ISADevice *isa;

    qemu_irq *cpu_exit_irq;

    int ppc_boot_device;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];



    sysctrl = g_malloc0(sizeof(sysctrl_t));



    linux_boot = (kernel_filename != NULL);



    /* init CPUs */

    if (cpu_model == NULL)

        cpu_model = "602";

    for (i = 0; i < smp_cpus; i++) {

        cpu = cpu_ppc_init(cpu_model);

        if (cpu == NULL) {

            fprintf(stderr, "Unable to find PowerPC CPU definition\n");

            exit(1);

        }

        env = &cpu->env;



        if (env->flags & POWERPC_FLAG_RTC_CLK) {

            /* POWER / PowerPC 601 RTC clock frequency is 7.8125 MHz */

            cpu_ppc_tb_init(env, 7812500UL);

        } else {

            /* Set time-base frequency to 100 Mhz */

            cpu_ppc_tb_init(env, 100UL * 1000UL * 1000UL);

        }

        qemu_register_reset(ppc_prep_reset, cpu);

    }



    /* allocate RAM */

    memory_region_init_ram(ram, NULL, "ppc_prep.ram", ram_size);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(sysmem, 0, ram);



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



    if (PPC_INPUT(env) != PPC_FLAGS_INPUT_6xx) {

        hw_error("Only 6xx bus is supported on PREP machine\n");

    }



    dev = qdev_create(NULL, "raven-pcihost");

    if (bios_name == NULL) {

        bios_name = BIOS_FILENAME;

    }

    qdev_prop_set_string(dev, "bios-name", bios_name);

    qdev_prop_set_uint32(dev, "elf-machine", ELF_MACHINE);

    pcihost = PCI_HOST_BRIDGE(dev);

    object_property_add_child(qdev_get_machine(), "raven", OBJECT(dev), NULL);

    qdev_init_nofail(dev);

    pci_bus = (PCIBus *)qdev_get_child_bus(dev, "pci.0");

    if (pci_bus == NULL) {

        fprintf(stderr, "Couldn't create PCI host controller.\n");

        exit(1);

    }

    sysctrl->contiguous_map_irq = qdev_get_gpio_in(dev, 0);



    /* PCI -> ISA bridge */

    pci = pci_create_simple(pci_bus, PCI_DEVFN(1, 0), "i82378");

    cpu_exit_irq = qemu_allocate_irqs(cpu_request_exit, NULL, 1);

    cpu = POWERPC_CPU(first_cpu);

    qdev_connect_gpio_out(&pci->qdev, 0,

                          cpu->env.irq_inputs[PPC6xx_INPUT_INT]);

    qdev_connect_gpio_out(&pci->qdev, 1, *cpu_exit_irq);

    sysbus_connect_irq(&pcihost->busdev, 0, qdev_get_gpio_in(&pci->qdev, 9));

    sysbus_connect_irq(&pcihost->busdev, 1, qdev_get_gpio_in(&pci->qdev, 11));

    sysbus_connect_irq(&pcihost->busdev, 2, qdev_get_gpio_in(&pci->qdev, 9));

    sysbus_connect_irq(&pcihost->busdev, 3, qdev_get_gpio_in(&pci->qdev, 11));

    isa_bus = ISA_BUS(qdev_get_child_bus(DEVICE(pci), "isa.0"));



    /* Super I/O (parallel + serial ports) */

    isa = isa_create(isa_bus, TYPE_PC87312);

    dev = DEVICE(isa);

    qdev_prop_set_uint8(dev, "config", 13); /* fdc, ser0, ser1, par0 */

    qdev_init_nofail(dev);



    /* init basic PC hardware */

    pci_vga_init(pci_bus);

    /* Open Hack'Ware hack: PCI BAR#0 is programmed to 0xf0000000.

     * While bios will access framebuffer at 0xf0000000, real physical

     * address is 0xf0000000 + 0xc0000000 (PCI memory base).

     * Alias the wrong memory accesses to the right place.

     */

    memory_region_init_alias(vga, NULL, "vga-alias", pci_address_space(pci),

                             0xf0000000, 0x1000000);

    memory_region_add_subregion_overlap(sysmem, 0xf0000000, vga, 10);



    nb_nics1 = nb_nics;

    if (nb_nics1 > NE2000_NB_MAX)

        nb_nics1 = NE2000_NB_MAX;

    for(i = 0; i < nb_nics1; i++) {

        if (nd_table[i].model == NULL) {

	    nd_table[i].model = g_strdup("ne2k_isa");

        }

        if (strcmp(nd_table[i].model, "ne2k_isa") == 0) {

            isa_ne2000_init(isa_bus, ne2000_io[i], ne2000_irq[i],

                            &nd_table[i]);

        } else {

            pci_nic_init_nofail(&nd_table[i], pci_bus, "ne2k_pci", NULL);

        }

    }



    ide_drive_get(hd, MAX_IDE_BUS);

    for(i = 0; i < MAX_IDE_BUS; i++) {

        isa_ide_init(isa_bus, ide_iobase[i], ide_iobase2[i], ide_irq[i],

                     hd[2 * i],

		     hd[2 * i + 1]);

    }

    isa_create_simple(isa_bus, "i8042");



    cpu = POWERPC_CPU(first_cpu);

    sysctrl->reset_irq = cpu->env.irq_inputs[PPC6xx_INPUT_HRESET];



    portio_list_init(port_list, NULL, prep_portio_list, sysctrl, "prep");

    portio_list_add(port_list, isa_address_space_io(isa), 0x0);



    /* PowerPC control and status register group */

#if 0

    memory_region_init_io(xcsr, NULL, &PPC_XCSR_ops, NULL, "ppc-xcsr", 0x1000);

    memory_region_add_subregion(sysmem, 0xFEFF0000, xcsr);

#endif



    if (usb_enabled(false)) {

        pci_create_simple(pci_bus, -1, "pci-ohci");

    }



    m48t59 = m48t59_init_isa(isa_bus, 0x0074, NVRAM_SIZE, 59);

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

}
