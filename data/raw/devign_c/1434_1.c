static void integratorcp_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    ARMCPU *cpu;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *ram_alias = g_new(MemoryRegion, 1);

    qemu_irq pic[32];

    DeviceState *dev;

    int i;



    if (!cpu_model) {

        cpu_model = "arm926";

    }

    cpu = cpu_arm_init(cpu_model);

    if (!cpu) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }



    memory_region_init_ram(ram, NULL, "integrator.ram", ram_size);

    vmstate_register_ram_global(ram);

    /* ??? On a real system the first 1Mb is mapped as SSRAM or boot flash.  */

    /* ??? RAM should repeat to fill physical memory space.  */

    /* SDRAM at address zero*/

    memory_region_add_subregion(address_space_mem, 0, ram);

    /* And again at address 0x80000000 */

    memory_region_init_alias(ram_alias, NULL, "ram.alias", ram, 0, ram_size);

    memory_region_add_subregion(address_space_mem, 0x80000000, ram_alias);



    dev = qdev_create(NULL, TYPE_INTEGRATOR_CM);

    qdev_prop_set_uint32(dev, "memsz", ram_size >> 20);

    qdev_init_nofail(dev);

    sysbus_mmio_map((SysBusDevice *)dev, 0, 0x10000000);



    dev = sysbus_create_varargs(TYPE_INTEGRATOR_PIC, 0x14000000,

                                qdev_get_gpio_in(DEVICE(cpu), ARM_CPU_IRQ),

                                qdev_get_gpio_in(DEVICE(cpu), ARM_CPU_FIQ),

                                NULL);

    for (i = 0; i < 32; i++) {

        pic[i] = qdev_get_gpio_in(dev, i);

    }

    sysbus_create_simple(TYPE_INTEGRATOR_PIC, 0xca000000, pic[26]);

    sysbus_create_varargs("integrator_pit", 0x13000000,

                          pic[5], pic[6], pic[7], NULL);

    sysbus_create_simple("pl031", 0x15000000, pic[8]);

    sysbus_create_simple("pl011", 0x16000000, pic[1]);

    sysbus_create_simple("pl011", 0x17000000, pic[2]);

    icp_control_init(0xcb000000);

    sysbus_create_simple("pl050_keyboard", 0x18000000, pic[3]);

    sysbus_create_simple("pl050_mouse", 0x19000000, pic[4]);


    sysbus_create_varargs("pl181", 0x1c000000, pic[23], pic[24], NULL);

    if (nd_table[0].used)

        smc91c111_init(&nd_table[0], 0xc8000000, pic[27]);



    sysbus_create_simple("pl110", 0xc0000000, pic[22]);



    integrator_binfo.ram_size = ram_size;

    integrator_binfo.kernel_filename = kernel_filename;

    integrator_binfo.kernel_cmdline = kernel_cmdline;

    integrator_binfo.initrd_filename = initrd_filename;

    arm_load_kernel(cpu, &integrator_binfo);

}