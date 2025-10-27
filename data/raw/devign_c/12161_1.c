static void calxeda_init(MachineState *machine, enum cxmachines machine_id)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    DeviceState *dev = NULL;

    SysBusDevice *busdev;

    qemu_irq pic[128];

    int n;

    qemu_irq cpu_irq[4];

    MemoryRegion *sysram;

    MemoryRegion *dram;

    MemoryRegion *sysmem;

    char *sysboot_filename;



    if (!cpu_model) {

        switch (machine_id) {

        case CALXEDA_HIGHBANK:

            cpu_model = "cortex-a9";

            break;

        case CALXEDA_MIDWAY:

            cpu_model = "cortex-a15";

            break;

        }

    }



    for (n = 0; n < smp_cpus; n++) {

        ObjectClass *oc = cpu_class_by_name(TYPE_ARM_CPU, cpu_model);

        Object *cpuobj;

        ARMCPU *cpu;

        Error *err = NULL;



        if (!oc) {

            error_report("Unable to find CPU definition");

            exit(1);

        }



        cpuobj = object_new(object_class_get_name(oc));

        cpu = ARM_CPU(cpuobj);



        /* By default A9 and A15 CPUs have EL3 enabled.  This board does not

         * currently support EL3 so the CPU EL3 property is disabled before

         * realization.

         */

        if (object_property_find(cpuobj, "has_el3", NULL)) {

            object_property_set_bool(cpuobj, false, "has_el3", &err);

            if (err) {

                error_report_err(err);

                exit(1);

            }

        }



        if (object_property_find(cpuobj, "reset-cbar", NULL)) {

            object_property_set_int(cpuobj, MPCORE_PERIPHBASE,

                                    "reset-cbar", &error_abort);

        }

        object_property_set_bool(cpuobj, true, "realized", &err);

        if (err) {

            error_report_err(err);

            exit(1);

        }

        cpu_irq[n] = qdev_get_gpio_in(DEVICE(cpu), ARM_CPU_IRQ);

    }



    sysmem = get_system_memory();

    dram = g_new(MemoryRegion, 1);

    memory_region_init_ram(dram, NULL, "highbank.dram", ram_size, &error_abort);

    /* SDRAM at address zero.  */

    memory_region_add_subregion(sysmem, 0, dram);



    sysram = g_new(MemoryRegion, 1);

    memory_region_init_ram(sysram, NULL, "highbank.sysram", 0x8000,

                           &error_abort);

    memory_region_add_subregion(sysmem, 0xfff88000, sysram);

    if (bios_name != NULL) {

        sysboot_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (sysboot_filename != NULL) {

            uint32_t filesize = get_image_size(sysboot_filename);

            if (load_image_targphys("sysram.bin", 0xfff88000, filesize) < 0) {

                hw_error("Unable to load %s\n", bios_name);

            }

            g_free(sysboot_filename);

        } else {

           hw_error("Unable to find %s\n", bios_name);

        }

    }



    switch (machine_id) {

    case CALXEDA_HIGHBANK:

        dev = qdev_create(NULL, "l2x0");

        qdev_init_nofail(dev);

        busdev = SYS_BUS_DEVICE(dev);

        sysbus_mmio_map(busdev, 0, 0xfff12000);



        dev = qdev_create(NULL, "a9mpcore_priv");

        break;

    case CALXEDA_MIDWAY:

        dev = qdev_create(NULL, "a15mpcore_priv");

        break;

    }

    qdev_prop_set_uint32(dev, "num-cpu", smp_cpus);

    qdev_prop_set_uint32(dev, "num-irq", NIRQ_GIC);

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(busdev, 0, MPCORE_PERIPHBASE);

    for (n = 0; n < smp_cpus; n++) {

        sysbus_connect_irq(busdev, n, cpu_irq[n]);

    }



    for (n = 0; n < 128; n++) {

        pic[n] = qdev_get_gpio_in(dev, n);

    }



    dev = qdev_create(NULL, "sp804");

    qdev_prop_set_uint32(dev, "freq0", 150000000);

    qdev_prop_set_uint32(dev, "freq1", 150000000);

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(busdev, 0, 0xfff34000);

    sysbus_connect_irq(busdev, 0, pic[18]);

    sysbus_create_simple("pl011", 0xfff36000, pic[20]);



    dev = qdev_create(NULL, "highbank-regs");

    qdev_init_nofail(dev);

    busdev = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(busdev, 0, 0xfff3c000);



    sysbus_create_simple("pl061", 0xfff30000, pic[14]);

    sysbus_create_simple("pl061", 0xfff31000, pic[15]);

    sysbus_create_simple("pl061", 0xfff32000, pic[16]);

    sysbus_create_simple("pl061", 0xfff33000, pic[17]);

    sysbus_create_simple("pl031", 0xfff35000, pic[19]);

    sysbus_create_simple("pl022", 0xfff39000, pic[23]);



    sysbus_create_simple("sysbus-ahci", 0xffe08000, pic[83]);



    if (nd_table[0].used) {

        qemu_check_nic_model(&nd_table[0], "xgmac");

        dev = qdev_create(NULL, "xgmac");

        qdev_set_nic_properties(dev, &nd_table[0]);

        qdev_init_nofail(dev);

        sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, 0xfff50000);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, pic[77]);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 1, pic[78]);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 2, pic[79]);



        qemu_check_nic_model(&nd_table[1], "xgmac");

        dev = qdev_create(NULL, "xgmac");

        qdev_set_nic_properties(dev, &nd_table[1]);

        qdev_init_nofail(dev);

        sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, 0xfff51000);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0, pic[80]);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 1, pic[81]);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 2, pic[82]);

    }



    highbank_binfo.ram_size = ram_size;

    highbank_binfo.kernel_filename = kernel_filename;

    highbank_binfo.kernel_cmdline = kernel_cmdline;

    highbank_binfo.initrd_filename = initrd_filename;

    /* highbank requires a dtb in order to boot, and the dtb will override

     * the board ID. The following value is ignored, so set it to -1 to be

     * clear that the value is meaningless.

     */

    highbank_binfo.board_id = -1;

    highbank_binfo.nb_cpus = smp_cpus;

    highbank_binfo.loader_start = 0;

    highbank_binfo.write_secondary_boot = hb_write_secondary;

    highbank_binfo.secondary_cpu_reset_hook = hb_reset_secondary;

    arm_load_kernel(ARM_CPU(first_cpu), &highbank_binfo);

}
