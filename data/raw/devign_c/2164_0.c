static void zynq_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    ObjectClass *cpu_oc;

    ARMCPU *cpu;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ext_ram = g_new(MemoryRegion, 1);

    MemoryRegion *ocm_ram = g_new(MemoryRegion, 1);

    DeviceState *dev;

    SysBusDevice *busdev;

    qemu_irq pic[64];

    NICInfo *nd;

    Error *err = NULL;

    int n;



    if (!cpu_model) {

        cpu_model = "cortex-a9";

    }

    cpu_oc = cpu_class_by_name(TYPE_ARM_CPU, cpu_model);



    cpu = ARM_CPU(object_new(object_class_get_name(cpu_oc)));



    object_property_set_int(OBJECT(cpu), MPCORE_PERIPHBASE, "reset-cbar", &err);

    if (err) {

        error_report("%s", error_get_pretty(err));

        exit(1);

    }

    object_property_set_bool(OBJECT(cpu), true, "realized", &err);

    if (err) {

        error_report("%s", error_get_pretty(err));

        exit(1);

    }



    /* max 2GB ram */

    if (ram_size > 0x80000000) {

        ram_size = 0x80000000;

    }



    /* DDR remapped to address zero.  */

    memory_region_init_ram(ext_ram, NULL, "zynq.ext_ram", ram_size);

    vmstate_register_ram_global(ext_ram);

    memory_region_add_subregion(address_space_mem, 0, ext_ram);



    /* 256K of on-chip memory */

    memory_region_init_ram(ocm_ram, NULL, "zynq.ocm_ram", 256 << 10);

    vmstate_register_ram_global(ocm_ram);

    memory_region_add_subregion(address_space_mem, 0xFFFC0000, ocm_ram);



    DriveInfo *dinfo = drive_get(IF_PFLASH, 0, 0);



    /* AMD */

    pflash_cfi02_register(0xe2000000, NULL, "zynq.pflash", FLASH_SIZE,

                          dinfo ? dinfo->bdrv : NULL, FLASH_SECTOR_SIZE,

                          FLASH_SIZE/FLASH_SECTOR_SIZE, 1,

                          1, 0x0066, 0x0022, 0x0000, 0x0000, 0x0555, 0x2aa,

                              0);



    dev = qdev_create(NULL, "xilinx,zynq_slcr");

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, 0xF8000000);



    dev = qdev_create(NULL, "a9mpcore_priv");

    qdev_prop_set_uint32(dev, "num-cpu", 1);

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(busdev, 0, MPCORE_PERIPHBASE);

    sysbus_connect_irq(busdev, 0,

                       qdev_get_gpio_in(DEVICE(cpu), ARM_CPU_IRQ));



    for (n = 0; n < 64; n++) {

        pic[n] = qdev_get_gpio_in(dev, n);

    }



    zynq_init_spi_flashes(0xE0006000, pic[58-IRQ_OFFSET], false);

    zynq_init_spi_flashes(0xE0007000, pic[81-IRQ_OFFSET], false);

    zynq_init_spi_flashes(0xE000D000, pic[51-IRQ_OFFSET], true);



    sysbus_create_simple("xlnx,ps7-usb", 0xE0002000, pic[53-IRQ_OFFSET]);

    sysbus_create_simple("xlnx,ps7-usb", 0xE0003000, pic[76-IRQ_OFFSET]);



    sysbus_create_simple("cadence_uart", 0xE0000000, pic[59-IRQ_OFFSET]);

    sysbus_create_simple("cadence_uart", 0xE0001000, pic[82-IRQ_OFFSET]);



    sysbus_create_varargs("cadence_ttc", 0xF8001000,

            pic[42-IRQ_OFFSET], pic[43-IRQ_OFFSET], pic[44-IRQ_OFFSET], NULL);

    sysbus_create_varargs("cadence_ttc", 0xF8002000,

            pic[69-IRQ_OFFSET], pic[70-IRQ_OFFSET], pic[71-IRQ_OFFSET], NULL);



    for (n = 0; n < nb_nics; n++) {

        nd = &nd_table[n];

        if (n == 0) {

            gem_init(nd, 0xE000B000, pic[54-IRQ_OFFSET]);

        } else if (n == 1) {

            gem_init(nd, 0xE000C000, pic[77-IRQ_OFFSET]);

        }

    }



    dev = qdev_create(NULL, "generic-sdhci");

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, 0xE0100000);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, pic[56-IRQ_OFFSET]);



    dev = qdev_create(NULL, "generic-sdhci");

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, 0xE0101000);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, pic[79-IRQ_OFFSET]);



    dev = qdev_create(NULL, "pl330");

    qdev_prop_set_uint8(dev, "num_chnls",  8);

    qdev_prop_set_uint8(dev, "num_periph_req",  4);

    qdev_prop_set_uint8(dev, "num_events",  16);



    qdev_prop_set_uint8(dev, "data_width",  64);

    qdev_prop_set_uint8(dev, "wr_cap",  8);

    qdev_prop_set_uint8(dev, "wr_q_dep",  16);

    qdev_prop_set_uint8(dev, "rd_cap",  8);

    qdev_prop_set_uint8(dev, "rd_q_dep",  16);

    qdev_prop_set_uint16(dev, "data_buffer_dep",  256);



    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(busdev, 0, 0xF8003000);

    sysbus_connect_irq(busdev, 0, pic[45-IRQ_OFFSET]); /* abort irq line */

    for (n = 0; n < 8; ++n) { /* event irqs */

        sysbus_connect_irq(busdev, n + 1, pic[dma_irqs[n] - IRQ_OFFSET]);

    }



    zynq_binfo.ram_size = ram_size;

    zynq_binfo.kernel_filename = kernel_filename;

    zynq_binfo.kernel_cmdline = kernel_cmdline;

    zynq_binfo.initrd_filename = initrd_filename;

    zynq_binfo.nb_cpus = 1;

    zynq_binfo.board_id = 0xd32;

    zynq_binfo.loader_start = 0;

    arm_load_kernel(ARM_CPU(first_cpu), &zynq_binfo);

}
