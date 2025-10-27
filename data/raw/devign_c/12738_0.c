static void tosa_init(ram_addr_t ram_size, int vga_ram_size,

                const char *boot_device,

                const char *kernel_filename, const char *kernel_cmdline,

                const char *initrd_filename, const char *cpu_model)

{

    struct pxa2xx_state_s *cpu;

    struct tc6393xb_s *tmio;

    struct scoop_info_s *scp0, *scp1;



    if (ram_size < (TOSA_RAM + TOSA_ROM + PXA2XX_INTERNAL_SIZE + TC6393XB_RAM)) {

        fprintf(stderr, "This platform requires %i bytes of memory\n",

                TOSA_RAM + TOSA_ROM + PXA2XX_INTERNAL_SIZE);

        exit(1);

    }



    if (!cpu_model)

        cpu_model = "pxa255";



    cpu = pxa255_init(tosa_binfo.ram_size);



    cpu_register_physical_memory(0, TOSA_ROM,

                    qemu_ram_alloc(TOSA_ROM) | IO_MEM_ROM);



    tmio = tc6393xb_init(0x10000000,

            pxa2xx_gpio_in_get(cpu->gpio)[TOSA_GPIO_TC6393XB_INT]);



    scp0 = scoop_init(cpu, 0, 0x08800000);

    scp1 = scoop_init(cpu, 1, 0x14800040);



    tosa_gpio_setup(cpu, scp0, scp1, tmio);



    tosa_microdrive_attach(cpu);



    tosa_tg_init(cpu);



    /* Setup initial (reset) machine state */

    cpu->env->regs[15] = tosa_binfo.loader_start;



    tosa_binfo.kernel_filename = kernel_filename;

    tosa_binfo.kernel_cmdline = kernel_cmdline;

    tosa_binfo.initrd_filename = initrd_filename;

    tosa_binfo.board_id = 0x208;

    arm_load_kernel(cpu->env, &tosa_binfo);

    sl_bootparam_write(SL_PXA_PARAM_BASE);

}
