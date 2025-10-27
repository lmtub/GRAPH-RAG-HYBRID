static void boston_mach_init(MachineState *machine)

{

    DeviceState *dev;

    BostonState *s;

    Error *err = NULL;

    const char *cpu_model;

    MemoryRegion *flash, *ddr, *ddr_low_alias, *lcd, *platreg;

    MemoryRegion *sys_mem = get_system_memory();

    XilinxPCIEHost *pcie2;

    PCIDevice *ahci;

    DriveInfo *hd[6];

    Chardev *chr;

    int fw_size, fit_err;

    bool is_64b;



    if ((machine->ram_size % G_BYTE) ||

        (machine->ram_size > (2 * G_BYTE))) {

        error_report("Memory size must be 1GB or 2GB");

        exit(1);

    }



    cpu_model = machine->cpu_model ?: "I6400";



    dev = qdev_create(NULL, TYPE_MIPS_BOSTON);

    qdev_init_nofail(dev);



    s = BOSTON(dev);

    s->mach = machine;

    s->cps = g_new0(MIPSCPSState, 1);



    if (!cpu_supports_cps_smp(cpu_model)) {

        error_report("Boston requires CPUs which support CPS");

        exit(1);

    }



    is_64b = cpu_supports_isa(cpu_model, ISA_MIPS64);



    object_initialize(s->cps, sizeof(MIPSCPSState), TYPE_MIPS_CPS);

    qdev_set_parent_bus(DEVICE(s->cps), sysbus_get_default());



    object_property_set_str(OBJECT(s->cps), cpu_model, "cpu-model", &err);

    object_property_set_int(OBJECT(s->cps), smp_cpus, "num-vp", &err);

    object_property_set_bool(OBJECT(s->cps), true, "realized", &err);



    if (err != NULL) {

        error_report("%s", error_get_pretty(err));

        exit(1);

    }



    sysbus_mmio_map_overlap(SYS_BUS_DEVICE(s->cps), 0, 0, 1);



    flash =  g_new(MemoryRegion, 1);

    memory_region_init_rom_device(flash, NULL, &boston_flash_ops, s,

                                  "boston.flash", 128 * M_BYTE, &err);

    memory_region_add_subregion_overlap(sys_mem, 0x18000000, flash, 0);



    ddr = g_new(MemoryRegion, 1);

    memory_region_allocate_system_memory(ddr, NULL, "boston.ddr",

                                         machine->ram_size);

    memory_region_add_subregion_overlap(sys_mem, 0x80000000, ddr, 0);



    ddr_low_alias = g_new(MemoryRegion, 1);

    memory_region_init_alias(ddr_low_alias, NULL, "boston_low.ddr",

                             ddr, 0, MIN(machine->ram_size, (256 * M_BYTE)));

    memory_region_add_subregion_overlap(sys_mem, 0, ddr_low_alias, 0);



    xilinx_pcie_init(sys_mem, 0,

                     0x10000000, 32 * M_BYTE,

                     0x40000000, 1 * G_BYTE,

                     get_cps_irq(s->cps, 2), false);



    xilinx_pcie_init(sys_mem, 1,

                     0x12000000, 32 * M_BYTE,

                     0x20000000, 512 * M_BYTE,

                     get_cps_irq(s->cps, 1), false);



    pcie2 = xilinx_pcie_init(sys_mem, 2,

                             0x14000000, 32 * M_BYTE,

                             0x16000000, 1 * M_BYTE,

                             get_cps_irq(s->cps, 0), true);



    platreg = g_new(MemoryRegion, 1);

    memory_region_init_io(platreg, NULL, &boston_platreg_ops, s,

                          "boston-platregs", 0x1000);

    memory_region_add_subregion_overlap(sys_mem, 0x17ffd000, platreg, 0);



    if (!serial_hds[0]) {

        serial_hds[0] = qemu_chr_new("serial0", "null");

    }



    s->uart = serial_mm_init(sys_mem, 0x17ffe000, 2,

                             get_cps_irq(s->cps, 3), 10000000,

                             serial_hds[0], DEVICE_NATIVE_ENDIAN);



    lcd = g_new(MemoryRegion, 1);

    memory_region_init_io(lcd, NULL, &boston_lcd_ops, s, "boston-lcd", 0x8);

    memory_region_add_subregion_overlap(sys_mem, 0x17fff000, lcd, 0);



    chr = qemu_chr_new("lcd", "vc:320x240");

    qemu_chr_fe_init(&s->lcd_display, chr, NULL);

    qemu_chr_fe_set_handlers(&s->lcd_display, NULL, NULL,

                             boston_lcd_event, s, NULL, true);



    ahci = pci_create_simple_multifunction(&PCI_BRIDGE(&pcie2->root)->sec_bus,

                                           PCI_DEVFN(0, 0),

                                           true, TYPE_ICH9_AHCI);

    g_assert(ARRAY_SIZE(hd) == ICH_AHCI(ahci)->ahci.ports);

    ide_drive_get(hd, ICH_AHCI(ahci)->ahci.ports);

    ahci_ide_create_devs(ahci, hd);



    if (machine->firmware) {

        fw_size = load_image_targphys(machine->firmware,

                                      0x1fc00000, 4 * M_BYTE);

        if (fw_size == -1) {

            error_printf("unable to load firmware image '%s'\n",

                          machine->firmware);

            exit(1);

        }

    } else if (machine->kernel_filename) {

        fit_err = load_fit(&boston_fit_loader, machine->kernel_filename, s);

        if (fit_err) {

            error_printf("unable to load FIT image\n");

            exit(1);

        }



        gen_firmware(memory_region_get_ram_ptr(flash) + 0x7c00000,

                     s->kernel_entry, s->fdt_base, is_64b);

    } else if (!qtest_enabled()) {

        error_printf("Please provide either a -kernel or -bios argument\n");

        exit(1);

    }

}
