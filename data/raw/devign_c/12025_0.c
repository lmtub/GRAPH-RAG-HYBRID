static void virtex_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    hwaddr initrd_base = 0;

    int initrd_size = 0;

    MemoryRegion *address_space_mem = get_system_memory();

    DeviceState *dev;

    PowerPCCPU *cpu;

    CPUPPCState *env;

    hwaddr ram_base = 0;

    DriveInfo *dinfo;

    MemoryRegion *phys_ram = g_new(MemoryRegion, 1);

    qemu_irq irq[32], *cpu_irq;

    int kernel_size;

    int i;



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "440-Xilinx";

    }



    cpu = ppc440_init_xilinx(&ram_size, 1, cpu_model, 400000000);

    env = &cpu->env;

    qemu_register_reset(main_cpu_reset, cpu);



    memory_region_allocate_system_memory(phys_ram, NULL, "ram", ram_size);

    memory_region_add_subregion(address_space_mem, ram_base, phys_ram);



    dinfo = drive_get(IF_PFLASH, 0, 0);

    pflash_cfi01_register(PFLASH_BASEADDR, NULL, "virtex.flash", FLASH_SIZE,

                          dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL,

                          (64 * 1024), FLASH_SIZE >> 16,

                          1, 0x89, 0x18, 0x0000, 0x0, 1);



    cpu_irq = (qemu_irq *) &env->irq_inputs[PPC40x_INPUT_INT];

    dev = qdev_create(NULL, "xlnx.xps-intc");

    qdev_prop_set_uint32(dev, "kind-of-intr", 0);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, INTC_BASEADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, cpu_irq[0]);

    for (i = 0; i < 32; i++) {

        irq[i] = qdev_get_gpio_in(dev, i);

    }



    serial_mm_init(address_space_mem, UART16550_BASEADDR, 2, irq[UART16550_IRQ],

                   115200, serial_hds[0], DEVICE_LITTLE_ENDIAN);



    /* 2 timers at irq 2 @ 62 Mhz.  */

    dev = qdev_create(NULL, "xlnx.xps-timer");

    qdev_prop_set_uint32(dev, "one-timer-only", 0);

    qdev_prop_set_uint32(dev, "clock-frequency", 62 * 1000000);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, TIMER_BASEADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, irq[TIMER_IRQ]);



    if (kernel_filename) {

        uint64_t entry, low, high;

        hwaddr boot_offset;



        /* Boots a kernel elf binary.  */

        kernel_size = load_elf(kernel_filename, NULL, NULL,

                               &entry, &low, &high, 1, ELF_MACHINE, 0);

        boot_info.bootstrap_pc = entry & 0x00ffffff;



        if (kernel_size < 0) {

            boot_offset = 0x1200000;

            /* If we failed loading ELF's try a raw image.  */

            kernel_size = load_image_targphys(kernel_filename,

                                              boot_offset,

                                              ram_size);

            boot_info.bootstrap_pc = boot_offset;

            high = boot_info.bootstrap_pc + kernel_size + 8192;

        }



        boot_info.ima_size = kernel_size;



        /* Load initrd. */

        if (machine->initrd_filename) {

            initrd_base = high = ROUND_UP(high, 4);

            initrd_size = load_image_targphys(machine->initrd_filename,

                                              high, ram_size - high);



            if (initrd_size < 0) {

                error_report("couldn't load ram disk '%s'",

                             machine->initrd_filename);

                exit(1);

            }

            high = ROUND_UP(high + initrd_size, 4);

        }



        /* Provide a device-tree.  */

        boot_info.fdt = high + (8192 * 2);

        boot_info.fdt &= ~8191;



        xilinx_load_device_tree(boot_info.fdt, ram_size,

                                initrd_base, initrd_size,

                                kernel_cmdline);

    }

    env->load_info = &boot_info;

}
