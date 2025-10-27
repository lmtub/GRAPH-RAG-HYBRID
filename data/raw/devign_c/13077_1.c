static void puv3_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *initrd_filename = machine->initrd_filename;

    CPUUniCore32State *env;

    UniCore32CPU *cpu;



    if (initrd_filename) {

        error_report("Please use kernel built-in initramdisk");

        exit(1);

    }



    if (!cpu_model) {

        cpu_model = "UniCore-II";

    }



    cpu = UNICORE32_CPU(cpu_generic_init(TYPE_UNICORE32_CPU, cpu_model));

    if (!cpu) {

        error_report("Unable to find CPU definition");

        exit(1);

    }

    env = &cpu->env;



    puv3_soc_init(env);

    puv3_board_init(env, ram_size);

    puv3_load_kernel(kernel_filename);

}
