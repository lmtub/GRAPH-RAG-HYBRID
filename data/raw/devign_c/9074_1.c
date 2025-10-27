static void leon3_generic_hw_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    SPARCCPU *cpu;

    CPUSPARCState   *env;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *prom = g_new(MemoryRegion, 1);

    int         ret;

    char       *filename;

    qemu_irq   *cpu_irqs = NULL;

    int         bios_size;

    int         prom_size;

    ResetData  *reset_info;



    /* Init CPU */

    if (!cpu_model) {

        cpu_model = "LEON3";

    }



    cpu = cpu_sparc_init(cpu_model);

    if (cpu == NULL) {

        fprintf(stderr, "qemu: Unable to find Sparc CPU definition\n");

        exit(1);

    }

    env = &cpu->env;



    cpu_sparc_set_id(env, 0);



    /* Reset data */

    reset_info        = g_malloc0(sizeof(ResetData));

    reset_info->cpu   = cpu;

    reset_info->sp    = 0x40000000 + ram_size;

    qemu_register_reset(main_cpu_reset, reset_info);



    /* Allocate IRQ manager */

    grlib_irqmp_create(0x80000200, env, &cpu_irqs, MAX_PILS, &leon3_set_pil_in);



    env->qemu_irq_ack = leon3_irq_manager;



    /* Allocate RAM */

    if ((uint64_t)ram_size > (1UL << 30)) {

        fprintf(stderr,

                "qemu: Too much memory for this machine: %d, maximum 1G\n",

                (unsigned int)(ram_size / (1024 * 1024)));

        exit(1);

    }



    memory_region_init_ram(ram, NULL, "leon3.ram", ram_size, &error_abort);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(address_space_mem, 0x40000000, ram);



    /* Allocate BIOS */

    prom_size = 8 * 1024 * 1024; /* 8Mb */

    memory_region_init_ram(prom, NULL, "Leon3.bios", prom_size, &error_abort);

    vmstate_register_ram_global(prom);

    memory_region_set_readonly(prom, true);

    memory_region_add_subregion(address_space_mem, 0x00000000, prom);



    /* Load boot prom */

    if (bios_name == NULL) {

        bios_name = PROM_FILENAME;

    }

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);



    bios_size = get_image_size(filename);



    if (bios_size > prom_size) {

        fprintf(stderr, "qemu: could not load prom '%s': file too big\n",

                filename);

        exit(1);

    }



    if (bios_size > 0) {

        ret = load_image_targphys(filename, 0x00000000, bios_size);

        if (ret < 0 || ret > prom_size) {

            fprintf(stderr, "qemu: could not load prom '%s'\n", filename);

            exit(1);

        }

    } else if (kernel_filename == NULL && !qtest_enabled()) {

        fprintf(stderr, "Can't read bios image %s\n", filename);

        exit(1);

    }




    /* Can directly load an application. */

    if (kernel_filename != NULL) {

        long     kernel_size;

        uint64_t entry;



        kernel_size = load_elf(kernel_filename, NULL, NULL, &entry, NULL, NULL,

                               1 /* big endian */, ELF_MACHINE, 0);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

        if (bios_size <= 0) {

            /* If there is no bios/monitor, start the application.  */

            env->pc = entry;

            env->npc = entry + 4;

            reset_info->entry = entry;

        }

    }



    /* Allocate timers */

    grlib_gptimer_create(0x80000300, 2, CPU_CLK, cpu_irqs, 6);



    /* Allocate uart */

    if (serial_hds[0]) {

        grlib_apbuart_create(0x80000100, serial_hds[0], cpu_irqs[3]);

    }

}