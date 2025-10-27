static void boston_mach_class_init(MachineClass *mc)

{

    mc->desc = "MIPS Boston";

    mc->init = boston_mach_init;

    mc->block_default_type = IF_IDE;

    mc->default_ram_size = 2 * G_BYTE;

    mc->max_cpus = 16;

}
