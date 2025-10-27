static void tosa_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *rom = g_new(MemoryRegion, 1);

    PXA2xxState *mpu;

    TC6393xbState *tmio;

    DeviceState *scp0, *scp1;



    if (!cpu_model)

        cpu_model = "pxa255";



    mpu = pxa255_init(address_space_mem, tosa_binfo.ram_size);



    memory_region_init_ram(rom, NULL, "tosa.rom", TOSA_ROM, &error_abort);

    vmstate_register_ram_global(rom);

    memory_region_set_readonly(rom, true);

    memory_region_add_subregion(address_space_mem, 0, rom);



    tmio = tc6393xb_init(address_space_mem, 0x10000000,

            qdev_get_gpio_in(mpu->gpio, TOSA_GPIO_TC6393XB_INT));



    scp0 = sysbus_create_simple("scoop", 0x08800000, NULL);

    scp1 = sysbus_create_simple("scoop", 0x14800040, NULL);



    tosa_gpio_setup(mpu, scp0, scp1, tmio);



    tosa_microdrive_attach(mpu);



    tosa_tg_init(mpu);



    tosa_binfo.kernel_filename = kernel_filename;

    tosa_binfo.kernel_cmdline = kernel_cmdline;

    tosa_binfo.initrd_filename = initrd_filename;

    tosa_binfo.board_id = 0x208;

    arm_load_kernel(mpu->cpu, &tosa_binfo);

    sl_bootparam_write(SL_PXA_PARAM_BASE);

}
