static void spitz_common_init(int ram_size, int vga_ram_size,

                DisplayState *ds, const char *kernel_filename,

                const char *kernel_cmdline, const char *initrd_filename,

                enum spitz_model_e model, int arm_id)

{

    uint32_t spitz_ram = 0x04000000;

    uint32_t spitz_rom = 0x00800000;

    struct pxa2xx_state_s *cpu;

    struct scoop_info_s *scp;



    cpu = pxa270_init(ds, (model == terrier) ? "c5" : "c0");



    /* Setup memory */

    if (ram_size < spitz_ram + spitz_rom) {

        fprintf(stderr, "This platform requires %i bytes of memory\n",

                        spitz_ram + spitz_rom);

        exit(1);

    }

    cpu_register_physical_memory(PXA2XX_RAM_BASE, spitz_ram, IO_MEM_RAM);



    sl_flash_register(cpu, (model == spitz) ? FLASH_128M : FLASH_1024M);



    cpu_register_physical_memory(0, spitz_rom, spitz_ram | IO_MEM_ROM);



    /* Setup peripherals */

    spitz_keyboard_register(cpu);



    spitz_ssp_attach(cpu);



    scp = spitz_scoop_init(cpu, (model == akita) ? 1 : 2);



    spitz_scoop_gpio_setup(cpu, scp, (model == akita) ? 1 : 2);



    spitz_gpio_setup(cpu, (model == akita) ? 1 : 2);



    if (model == terrier)

        /* A 6.0 GB microdrive is permanently sitting in CF slot 0.  */

        spitz_microdrive_attach(cpu);

    else if (model != akita)

        /* A 4.0 GB microdrive is permanently sitting in CF slot 0.  */

        spitz_microdrive_attach(cpu);



    /* Setup initial (reset) machine state */

    cpu->env->regs[15] = PXA2XX_RAM_BASE;



    arm_load_kernel(cpu->env, ram_size, kernel_filename, kernel_cmdline,

                    initrd_filename, arm_id, PXA2XX_RAM_BASE);

    sl_bootparam_write(SL_PXA_PARAM_BASE - PXA2XX_RAM_BASE);

}
