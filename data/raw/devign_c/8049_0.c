static void cubieboard_init(QEMUMachineInitArgs *args)

{

    CubieBoardState *s = g_new(CubieBoardState, 1);

    Error *err = NULL;



    s->a10 = AW_A10(object_new(TYPE_AW_A10));

    object_property_set_bool(OBJECT(s->a10), true, "realized", &err);

    if (err != NULL) {

        error_report("Couldn't realize Allwinner A10: %s\n",

                error_get_pretty(err));

        exit(1);

    }



    memory_region_init_ram(&s->sdram, NULL, "cubieboard.ram", args->ram_size);

    vmstate_register_ram_global(&s->sdram);

    memory_region_add_subregion(get_system_memory(), AW_A10_SDRAM_BASE,

                                &s->sdram);



    cubieboard_binfo.ram_size = args->ram_size;

    cubieboard_binfo.kernel_filename = args->kernel_filename;

    cubieboard_binfo.kernel_cmdline = args->kernel_cmdline;

    arm_load_kernel(&s->a10->cpu, &cubieboard_binfo);

}
