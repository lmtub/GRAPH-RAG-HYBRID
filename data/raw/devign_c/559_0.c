petalogix_ml605_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    MemoryRegion *address_space_mem = get_system_memory();

    DeviceState *dev, *dma, *eth0;

    Object *peer;

    MicroBlazeCPU *cpu;

    SysBusDevice *busdev;

    CPUMBState *env;

    DriveInfo *dinfo;

    int i;

    hwaddr ddr_base = MEMORY_BASEADDR;

    MemoryRegion *phys_lmb_bram = g_new(MemoryRegion, 1);

    MemoryRegion *phys_ram = g_new(MemoryRegion, 1);

    qemu_irq irq[32], *cpu_irq;



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "microblaze";

    }

    cpu = cpu_mb_init(cpu_model);

    env = &cpu->env;



    /* Attach emulated BRAM through the LMB.  */

    memory_region_init_ram(phys_lmb_bram, "petalogix_ml605.lmb_bram",

                           LMB_BRAM_SIZE);

    vmstate_register_ram_global(phys_lmb_bram);

    memory_region_add_subregion(address_space_mem, 0x00000000, phys_lmb_bram);



    memory_region_init_ram(phys_ram, "petalogix_ml605.ram", ram_size);

    vmstate_register_ram_global(phys_ram);

    memory_region_add_subregion(address_space_mem, ddr_base, phys_ram);



    dinfo = drive_get(IF_PFLASH, 0, 0);

    /* 5th parameter 2 means bank-width

     * 10th paremeter 0 means little-endian */

    pflash_cfi01_register(FLASH_BASEADDR,

                          NULL, "petalogix_ml605.flash", FLASH_SIZE,

                          dinfo ? dinfo->bdrv : NULL, (64 * 1024),

                          FLASH_SIZE >> 16,

                          2, 0x89, 0x18, 0x0000, 0x0, 0);





    cpu_irq = microblaze_pic_init_cpu(env);

    dev = xilinx_intc_create(INTC_BASEADDR, cpu_irq[0], 4);

    for (i = 0; i < 32; i++) {

        irq[i] = qdev_get_gpio_in(dev, i);

    }



    serial_mm_init(address_space_mem, UART16550_BASEADDR + 0x1000, 2,

                   irq[5], 115200, serial_hds[0], DEVICE_LITTLE_ENDIAN);



    /* 2 timers at irq 2 @ 100 Mhz.  */

    xilinx_timer_create(TIMER_BASEADDR, irq[2], 0, 100 * 1000000);



    /* axi ethernet and dma initialization. */

    qemu_check_nic_model(&nd_table[0], "xlnx.axi-ethernet");

    eth0 = qdev_create(NULL, "xlnx.axi-ethernet");

    dma = qdev_create(NULL, "xlnx.axi-dma");



    /* FIXME: attach to the sysbus instead */

    object_property_add_child(qdev_get_machine(), "xilinx-eth", OBJECT(eth0),

                              NULL);

    object_property_add_child(qdev_get_machine(), "xilinx-dma", OBJECT(dma),

                              NULL);



    peer = object_property_get_link(OBJECT(dma),

                                    "axistream-connected-target", NULL);

    xilinx_axiethernet_init(eth0, &nd_table[0], STREAM_SLAVE(peer),

                            0x82780000, irq[3], 0x1000, 0x1000);



    peer = object_property_get_link(OBJECT(eth0),

                                    "axistream-connected-target", NULL);

    xilinx_axidma_init(dma, STREAM_SLAVE(peer), 0x84600000, irq[1], irq[0],

                       100 * 1000000);



    {

        SSIBus *spi;



        dev = qdev_create(NULL, "xlnx.xps-spi");

        qdev_prop_set_uint8(dev, "num-ss-bits", NUM_SPI_FLASHES);

        qdev_init_nofail(dev);

        busdev = SYS_BUS_DEVICE(dev);

        sysbus_mmio_map(busdev, 0, 0x40a00000);

        sysbus_connect_irq(busdev, 0, irq[4]);



        spi = (SSIBus *)qdev_get_child_bus(dev, "spi");



        for (i = 0; i < NUM_SPI_FLASHES; i++) {

            qemu_irq cs_line;



            dev = ssi_create_slave(spi, "n25q128");

            cs_line = qdev_get_gpio_in(dev, 0);

            sysbus_connect_irq(busdev, i+1, cs_line);

        }

    }



    microblaze_load_kernel(cpu, ddr_base, ram_size, BINARY_DEVICE_TREE_FILE,

                                                            machine_cpu_reset);



}
