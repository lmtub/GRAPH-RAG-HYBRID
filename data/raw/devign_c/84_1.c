static void lm32_evr_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    LM32CPU *cpu;

    CPULM32State *env;

    DriveInfo *dinfo;

    MemoryRegion *address_space_mem =  get_system_memory();

    MemoryRegion *phys_ram = g_new(MemoryRegion, 1);

    qemu_irq irq[32];

    ResetInfo *reset_info;

    int i;



    /* memory map */

    hwaddr flash_base  = 0x04000000;

    size_t flash_sector_size       = 256 * 1024;

    size_t flash_size              = 32 * 1024 * 1024;

    hwaddr ram_base    = 0x08000000;

    size_t ram_size                = 64 * 1024 * 1024;

    hwaddr timer0_base = 0x80002000;

    hwaddr uart0_base  = 0x80006000;

    hwaddr timer1_base = 0x8000a000;

    int uart0_irq                  = 0;

    int timer0_irq                 = 1;

    int timer1_irq                 = 3;



    reset_info = g_malloc0(sizeof(ResetInfo));



    if (cpu_model == NULL) {

        cpu_model = "lm32-full";

    }

    cpu = LM32_CPU(cpu_generic_init(TYPE_LM32_CPU, cpu_model));

    if (cpu == NULL) {

        fprintf(stderr, "qemu: unable to find CPU '%s'\n", cpu_model);

        exit(1);

    }



    env = &cpu->env;

    reset_info->cpu = cpu;



    reset_info->flash_base = flash_base;



    memory_region_allocate_system_memory(phys_ram, NULL, "lm32_evr.sdram",

                                         ram_size);

    memory_region_add_subregion(address_space_mem, ram_base, phys_ram);



    dinfo = drive_get(IF_PFLASH, 0, 0);

    /* Spansion S29NS128P */

    pflash_cfi02_register(flash_base, NULL, "lm32_evr.flash", flash_size,

                          dinfo ? blk_by_legacy_dinfo(dinfo) : NULL,

                          flash_sector_size, flash_size / flash_sector_size,

                          1, 2, 0x01, 0x7e, 0x43, 0x00, 0x555, 0x2aa, 1);



    /* create irq lines */

    env->pic_state = lm32_pic_init(qemu_allocate_irq(cpu_irq_handler, cpu, 0));

    for (i = 0; i < 32; i++) {

        irq[i] = qdev_get_gpio_in(env->pic_state, i);

    }



    lm32_uart_create(uart0_base, irq[uart0_irq], serial_hds[0]);

    sysbus_create_simple("lm32-timer", timer0_base, irq[timer0_irq]);

    sysbus_create_simple("lm32-timer", timer1_base, irq[timer1_irq]);



    /* make sure juart isn't the first chardev */

    env->juart_state = lm32_juart_init(serial_hds[1]);



    reset_info->bootstrap_pc = flash_base;



    if (kernel_filename) {

        uint64_t entry;

        int kernel_size;



        kernel_size = load_elf(kernel_filename, NULL, NULL, &entry, NULL, NULL,

                               1, EM_LATTICEMICO32, 0, 0);

        reset_info->bootstrap_pc = entry;



        if (kernel_size < 0) {

            kernel_size = load_image_targphys(kernel_filename, ram_base,

                                              ram_size);

            reset_info->bootstrap_pc = ram_base;

        }



        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

    }



    qemu_register_reset(main_cpu_reset, reset_info);

}
