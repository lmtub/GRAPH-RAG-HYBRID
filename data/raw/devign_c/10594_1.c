void ppce500_init(MachineState *machine, PPCE500Params *params)

{

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    PCIBus *pci_bus;

    CPUPPCState *env = NULL;

    uint64_t loadaddr;

    hwaddr kernel_base = -1LL;

    int kernel_size = 0;

    hwaddr dt_base = 0;

    hwaddr initrd_base = 0;

    int initrd_size = 0;

    hwaddr cur_base = 0;

    char *filename;

    hwaddr bios_entry = 0;

    target_long bios_size;

    struct boot_info *boot_info;

    int dt_size;

    int i;

    /* irq num for pin INTA, INTB, INTC and INTD is 1, 2, 3 and

     * 4 respectively */

    unsigned int pci_irq_nrs[PCI_NUM_PINS] = {1, 2, 3, 4};

    qemu_irq **irqs, *mpic;

    DeviceState *dev;

    CPUPPCState *firstenv = NULL;

    MemoryRegion *ccsr_addr_space;

    SysBusDevice *s;

    PPCE500CCSRState *ccsr;



    /* Setup CPUs */

    if (machine->cpu_model == NULL) {

        machine->cpu_model = "e500v2_v30";

    }



    irqs = g_malloc0(smp_cpus * sizeof(qemu_irq *));

    irqs[0] = g_malloc0(smp_cpus * sizeof(qemu_irq) * OPENPIC_OUTPUT_NB);

    for (i = 0; i < smp_cpus; i++) {

        PowerPCCPU *cpu;

        CPUState *cs;

        qemu_irq *input;



        cpu = cpu_ppc_init(machine->cpu_model);

        if (cpu == NULL) {

            fprintf(stderr, "Unable to initialize CPU!\n");

            exit(1);

        }

        env = &cpu->env;

        cs = CPU(cpu);



        if (!firstenv) {

            firstenv = env;

        }



        irqs[i] = irqs[0] + (i * OPENPIC_OUTPUT_NB);

        input = (qemu_irq *)env->irq_inputs;

        irqs[i][OPENPIC_OUTPUT_INT] = input[PPCE500_INPUT_INT];

        irqs[i][OPENPIC_OUTPUT_CINT] = input[PPCE500_INPUT_CINT];

        env->spr_cb[SPR_BOOKE_PIR].default_value = cs->cpu_index = i;

        env->mpic_iack = params->ccsrbar_base +

                         MPC8544_MPIC_REGS_OFFSET + 0xa0;



        ppc_booke_timers_init(cpu, 400000000, PPC_TIMER_E500);



        /* Register reset handler */

        if (!i) {

            /* Primary CPU */

            struct boot_info *boot_info;

            boot_info = g_malloc0(sizeof(struct boot_info));

            qemu_register_reset(ppce500_cpu_reset, cpu);

            env->load_info = boot_info;

        } else {

            /* Secondary CPUs */

            qemu_register_reset(ppce500_cpu_reset_sec, cpu);

        }

    }



    env = firstenv;



    /* Fixup Memory size on a alignment boundary */

    ram_size &= ~(RAM_SIZES_ALIGN - 1);

    machine->ram_size = ram_size;



    /* Register Memory */

    memory_region_allocate_system_memory(ram, NULL, "mpc8544ds.ram", ram_size);

    memory_region_add_subregion(address_space_mem, 0, ram);



    dev = qdev_create(NULL, "e500-ccsr");

    object_property_add_child(qdev_get_machine(), "e500-ccsr",

                              OBJECT(dev), NULL);

    qdev_init_nofail(dev);

    ccsr = CCSR(dev);

    ccsr_addr_space = &ccsr->ccsr_space;

    memory_region_add_subregion(address_space_mem, params->ccsrbar_base,

                                ccsr_addr_space);



    mpic = ppce500_init_mpic(params, ccsr_addr_space, irqs);



    /* Serial */

    if (serial_hds[0]) {

        serial_mm_init(ccsr_addr_space, MPC8544_SERIAL0_REGS_OFFSET,

                       0, mpic[42], 399193,

                       serial_hds[0], DEVICE_BIG_ENDIAN);

    }



    if (serial_hds[1]) {

        serial_mm_init(ccsr_addr_space, MPC8544_SERIAL1_REGS_OFFSET,

                       0, mpic[42], 399193,

                       serial_hds[1], DEVICE_BIG_ENDIAN);

    }



    /* General Utility device */

    dev = qdev_create(NULL, "mpc8544-guts");

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    memory_region_add_subregion(ccsr_addr_space, MPC8544_UTIL_OFFSET,

                                sysbus_mmio_get_region(s, 0));



    /* PCI */

    dev = qdev_create(NULL, "e500-pcihost");

    qdev_prop_set_uint32(dev, "first_slot", params->pci_first_slot);

    qdev_prop_set_uint32(dev, "first_pin_irq", pci_irq_nrs[0]);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    for (i = 0; i < PCI_NUM_PINS; i++) {

        sysbus_connect_irq(s, i, mpic[pci_irq_nrs[i]]);

    }



    memory_region_add_subregion(ccsr_addr_space, MPC8544_PCI_REGS_OFFSET,

                                sysbus_mmio_get_region(s, 0));



    pci_bus = (PCIBus *)qdev_get_child_bus(dev, "pci.0");

    if (!pci_bus)

        printf("couldn't create PCI controller!\n");



    if (pci_bus) {

        /* Register network interfaces. */

        for (i = 0; i < nb_nics; i++) {

            pci_nic_init_nofail(&nd_table[i], pci_bus, "virtio", NULL);

        }

    }



    /* Register spinning region */

    sysbus_create_simple("e500-spin", params->spin_base, NULL);



    if (cur_base < (32 * 1024 * 1024)) {

        /* u-boot occupies memory up to 32MB, so load blobs above */

        cur_base = (32 * 1024 * 1024);

    }



    if (params->has_mpc8xxx_gpio) {

        qemu_irq poweroff_irq;



        dev = qdev_create(NULL, "mpc8xxx_gpio");

        s = SYS_BUS_DEVICE(dev);

        qdev_init_nofail(dev);

        sysbus_connect_irq(s, 0, mpic[MPC8XXX_GPIO_IRQ]);

        memory_region_add_subregion(ccsr_addr_space, MPC8XXX_GPIO_OFFSET,

                                    sysbus_mmio_get_region(s, 0));



        /* Power Off GPIO at Pin 0 */

        poweroff_irq = qemu_allocate_irq(ppce500_power_off, NULL, 0);

        qdev_connect_gpio_out(dev, 0, poweroff_irq);

    }



    /* Platform Bus Device */

    if (params->has_platform_bus) {

        dev = qdev_create(NULL, TYPE_PLATFORM_BUS_DEVICE);

        dev->id = TYPE_PLATFORM_BUS_DEVICE;

        qdev_prop_set_uint32(dev, "num_irqs", params->platform_bus_num_irqs);

        qdev_prop_set_uint32(dev, "mmio_size", params->platform_bus_size);

        qdev_init_nofail(dev);

        s = SYS_BUS_DEVICE(dev);



        for (i = 0; i < params->platform_bus_num_irqs; i++) {

            int irqn = params->platform_bus_first_irq + i;

            sysbus_connect_irq(s, i, mpic[irqn]);

        }



        memory_region_add_subregion(address_space_mem,

                                    params->platform_bus_base,

                                    sysbus_mmio_get_region(s, 0));

    }



    /* Load kernel. */

    if (machine->kernel_filename) {

        kernel_base = cur_base;

        kernel_size = load_image_targphys(machine->kernel_filename,

                                          cur_base,

                                          ram_size - cur_base);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    machine->kernel_filename);

            exit(1);

        }



        cur_base += kernel_size;

    }



    /* Load initrd. */

    if (machine->initrd_filename) {

        initrd_base = (cur_base + INITRD_LOAD_PAD) & ~INITRD_PAD_MASK;

        initrd_size = load_image_targphys(machine->initrd_filename, initrd_base,

                                          ram_size - initrd_base);



        if (initrd_size < 0) {

            fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                    machine->initrd_filename);

            exit(1);

        }



        cur_base = initrd_base + initrd_size;

    }



    /*

     * Smart firmware defaults ahead!

     *

     * We follow the following table to select which payload we execute.

     *

     *  -kernel | -bios | payload

     * ---------+-------+---------

     *     N    |   Y   | u-boot

     *     N    |   N   | u-boot

     *     Y    |   Y   | u-boot

     *     Y    |   N   | kernel

     *

     * This ensures backwards compatibility with how we used to expose

     * -kernel to users but allows them to run through u-boot as well.

     */

    if (bios_name == NULL) {

        if (machine->kernel_filename) {

            bios_name = machine->kernel_filename;

        } else {

            bios_name = "u-boot.e500";

        }

    }

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);



    bios_size = load_elf(filename, NULL, NULL, &bios_entry, &loadaddr, NULL,

                         1, ELF_MACHINE, 0);

    if (bios_size < 0) {

        /*

         * Hrm. No ELF image? Try a uImage, maybe someone is giving us an

         * ePAPR compliant kernel

         */

        kernel_size = load_uimage(filename, &bios_entry, &loadaddr, NULL,

                                  NULL, NULL);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load firmware '%s'\n", filename);

            exit(1);

        }

    }



    /* Reserve space for dtb */

    dt_base = (loadaddr + bios_size + DTC_LOAD_PAD) & ~DTC_PAD_MASK;



    dt_size = ppce500_prep_device_tree(machine, params, dt_base,

                                       initrd_base, initrd_size,

                                       kernel_base, kernel_size);

    if (dt_size < 0) {

        fprintf(stderr, "couldn't load device tree\n");

        exit(1);

    }

    assert(dt_size < DTB_MAX_SIZE);



    boot_info = env->load_info;

    boot_info->entry = bios_entry;

    boot_info->dt_base = dt_base;

    boot_info->dt_size = dt_size;



    if (kvm_enabled()) {

        kvmppc_init();

    }

}
