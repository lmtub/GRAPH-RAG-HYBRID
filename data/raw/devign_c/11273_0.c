static void verdex_init(ram_addr_t ram_size, int vga_ram_size,

                const char *boot_device,

                const char *kernel_filename, const char *kernel_cmdline,

                const char *initrd_filename, const char *cpu_model)

{

    struct pxa2xx_state_s *cpu;

    int index;



    uint32_t verdex_rom = 0x02000000;

    uint32_t verdex_ram = 0x10000000;



    if (ram_size < (verdex_ram + verdex_rom + PXA2XX_INTERNAL_SIZE)) {

        fprintf(stderr, "This platform requires %i bytes of memory\n",

                verdex_ram + verdex_rom + PXA2XX_INTERNAL_SIZE);

        exit(1);

    }



    cpu = pxa270_init(verdex_ram, cpu_model ?: "pxa270-c0");



    index = drive_get_index(IF_PFLASH, 0, 0);

    if (index == -1) {

        fprintf(stderr, "A flash image must be given with the "

                "'pflash' parameter\n");

        exit(1);

    }



    if (!pflash_cfi01_register(0x00000000, qemu_ram_alloc(verdex_rom),

            drives_table[index].bdrv, sector_len, verdex_rom / sector_len,

            2, 0, 0, 0, 0)) {

        fprintf(stderr, "qemu: Error registering flash memory.\n");

        exit(1);

    }



    cpu->env->regs[15] = 0x00000000;



    /* Interrupt line of NIC is connected to GPIO line 99 */

    smc91c111_init(&nd_table[0], 0x04000300,

                    pxa2xx_gpio_in_get(cpu->gpio)[99]);

}
