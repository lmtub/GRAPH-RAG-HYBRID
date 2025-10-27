static void r2d_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    SuperHCPU *cpu;

    CPUSH4State *env;

    ResetData *reset_info;

    struct SH7750State *s;

    MemoryRegion *sdram = g_new(MemoryRegion, 1);

    qemu_irq *irq;

    DriveInfo *dinfo;

    int i;

    DeviceState *dev;

    SysBusDevice *busdev;

    MemoryRegion *address_space_mem = get_system_memory();

    PCIBus *pci_bus;



    if (cpu_model == NULL) {

        cpu_model = "SH7751R";

    }



    cpu = cpu_sh4_init(cpu_model);

    if (cpu == NULL) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    reset_info = g_malloc0(sizeof(ResetData));

    reset_info->cpu = cpu;

    reset_info->vector = env->pc;

    qemu_register_reset(main_cpu_reset, reset_info);



    /* Allocate memory space */

    memory_region_init_ram(sdram, NULL, "r2d.sdram", SDRAM_SIZE, &error_abort);

    vmstate_register_ram_global(sdram);

    memory_region_add_subregion(address_space_mem, SDRAM_BASE, sdram);

    /* Register peripherals */

    s = sh7750_init(cpu, address_space_mem);

    irq = r2d_fpga_init(address_space_mem, 0x04000000, sh7750_irl(s));



    dev = qdev_create(NULL, "sh_pci");

    busdev = SYS_BUS_DEVICE(dev);

    qdev_init_nofail(dev);

    pci_bus = PCI_BUS(qdev_get_child_bus(dev, "pci"));

    sysbus_mmio_map(busdev, 0, P4ADDR(0x1e200000));

    sysbus_mmio_map(busdev, 1, A7ADDR(0x1e200000));

    sysbus_connect_irq(busdev, 0, irq[PCI_INTA]);

    sysbus_connect_irq(busdev, 1, irq[PCI_INTB]);

    sysbus_connect_irq(busdev, 2, irq[PCI_INTC]);

    sysbus_connect_irq(busdev, 3, irq[PCI_INTD]);



    sm501_init(address_space_mem, 0x10000000, SM501_VRAM_SIZE,

               irq[SM501], serial_hds[2]);



    /* onboard CF (True IDE mode, Master only). */

    dinfo = drive_get(IF_IDE, 0, 0);

    dev = qdev_create(NULL, "mmio-ide");

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_connect_irq(busdev, 0, irq[CF_IDE]);

    qdev_prop_set_uint32(dev, "shift", 1);

    qdev_init_nofail(dev);

    sysbus_mmio_map(busdev, 0, 0x14001000);

    sysbus_mmio_map(busdev, 1, 0x1400080c);

    mmio_ide_init_drives(dev, dinfo, NULL);



    /* onboard flash memory */

    dinfo = drive_get(IF_PFLASH, 0, 0);

    pflash_cfi02_register(0x0, NULL, "r2d.flash", FLASH_SIZE,

                          dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL,

                          (16 * 1024), FLASH_SIZE >> 16,

                          1, 4, 0x0000, 0x0000, 0x0000, 0x0000,

                          0x555, 0x2aa, 0);



    /* NIC: rtl8139 on-board, and 2 slots. */

    for (i = 0; i < nb_nics; i++)

        pci_nic_init_nofail(&nd_table[i], pci_bus,

                            "rtl8139", i==0 ? "2" : NULL);



    /* USB keyboard */

    usbdevice_create("keyboard");



    /* Todo: register on board registers */

    memset(&boot_params, 0, sizeof(boot_params));



    if (kernel_filename) {

        int kernel_size;



        kernel_size = load_image_targphys(kernel_filename,

                                          SDRAM_BASE + LINUX_LOAD_OFFSET,

                                          INITRD_LOAD_OFFSET - LINUX_LOAD_OFFSET);

        if (kernel_size < 0) {

          fprintf(stderr, "qemu: could not load kernel '%s'\n", kernel_filename);

          exit(1);

        }



        /* initialization which should be done by firmware */

        stl_phys(&address_space_memory, SH7750_BCR1, 1<<3); /* cs3 SDRAM */

        stw_phys(&address_space_memory, SH7750_BCR2, 3<<(3*2)); /* cs3 32bit */

        reset_info->vector = (SDRAM_BASE + LINUX_LOAD_OFFSET) | 0xa0000000; /* Start from P2 area */

    }



    if (initrd_filename) {

        int initrd_size;



        initrd_size = load_image_targphys(initrd_filename,

                                          SDRAM_BASE + INITRD_LOAD_OFFSET,

                                          SDRAM_SIZE - INITRD_LOAD_OFFSET);



        if (initrd_size < 0) {

          fprintf(stderr, "qemu: could not load initrd '%s'\n", initrd_filename);

          exit(1);

        }



        /* initialization which should be done by firmware */

        boot_params.loader_type = 1;

        boot_params.initrd_start = INITRD_LOAD_OFFSET;

        boot_params.initrd_size = initrd_size;

    }



    if (kernel_cmdline) {

        /* I see no evidence that this .kernel_cmdline buffer requires

           NUL-termination, so using strncpy should be ok. */

        strncpy(boot_params.kernel_cmdline, kernel_cmdline,

                sizeof(boot_params.kernel_cmdline));

    }



    rom_add_blob_fixed("boot_params", &boot_params, sizeof(boot_params),

                       SDRAM_BASE + BOOT_PARAMS_OFFSET);

}
