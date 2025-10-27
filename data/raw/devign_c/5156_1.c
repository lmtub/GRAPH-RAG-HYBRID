petalogix_s3adsp1800_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    DeviceState *dev;

    MicroBlazeCPU *cpu;

    DriveInfo *dinfo;

    int i;

    hwaddr ddr_base = MEMORY_BASEADDR;

    MemoryRegion *phys_lmb_bram = g_new(MemoryRegion, 1);

    MemoryRegion *phys_ram = g_new(MemoryRegion, 1);

    qemu_irq irq[32];

    MemoryRegion *sysmem = get_system_memory();



    cpu = MICROBLAZE_CPU(object_new(TYPE_MICROBLAZE_CPU));

    object_property_set_bool(OBJECT(cpu), true, "realized", &error_abort);



    /* Attach emulated BRAM through the LMB.  */

    memory_region_init_ram(phys_lmb_bram, NULL,

                           "petalogix_s3adsp1800.lmb_bram", LMB_BRAM_SIZE,

                           &error_abort);

    vmstate_register_ram_global(phys_lmb_bram);

    memory_region_add_subregion(sysmem, 0x00000000, phys_lmb_bram);



    memory_region_init_ram(phys_ram, NULL, "petalogix_s3adsp1800.ram",

                           ram_size, &error_abort);

    vmstate_register_ram_global(phys_ram);

    memory_region_add_subregion(sysmem, ddr_base, phys_ram);



    dinfo = drive_get(IF_PFLASH, 0, 0);

    pflash_cfi01_register(FLASH_BASEADDR,

                          NULL, "petalogix_s3adsp1800.flash", FLASH_SIZE,

                          dinfo ? blk_by_legacy_dinfo(dinfo) : NULL,

                          (64 * 1024), FLASH_SIZE >> 16,

                          1, 0x89, 0x18, 0x0000, 0x0, 1);



    dev = qdev_create(NULL, "xlnx.xps-intc");

    qdev_prop_set_uint32(dev, "kind-of-intr",

                         1 << ETHLITE_IRQ | 1 << UARTLITE_IRQ);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, INTC_BASEADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0,

                       qdev_get_gpio_in(DEVICE(cpu), MB_CPU_IRQ));

    for (i = 0; i < 32; i++) {

        irq[i] = qdev_get_gpio_in(dev, i);

    }



    sysbus_create_simple("xlnx.xps-uartlite", UARTLITE_BASEADDR,

                         irq[UARTLITE_IRQ]);



    /* 2 timers at irq 2 @ 62 Mhz.  */

    dev = qdev_create(NULL, "xlnx.xps-timer");

    qdev_prop_set_uint32(dev, "one-timer-only", 0);

    qdev_prop_set_uint32(dev, "clock-frequency", 62 * 1000000);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, TIMER_BASEADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, irq[TIMER_IRQ]);



    qemu_check_nic_model(&nd_table[0], "xlnx.xps-ethernetlite");

    dev = qdev_create(NULL, "xlnx.xps-ethernetlite");

    qdev_set_nic_properties(dev, &nd_table[0]);

    qdev_prop_set_uint32(dev, "tx-ping-pong", 0);

    qdev_prop_set_uint32(dev, "rx-ping-pong", 0);

    qdev_init_nofail(dev);

    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, ETHLITE_BASEADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, irq[ETHLITE_IRQ]);



    microblaze_load_kernel(cpu, ddr_base, ram_size,

                           machine->initrd_filename,

                           BINARY_DEVICE_TREE_FILE,

                           NULL);

}
