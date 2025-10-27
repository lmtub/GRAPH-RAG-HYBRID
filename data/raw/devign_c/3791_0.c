static void mainstone_common_init(ram_addr_t ram_size, int vga_ram_size,

                const char *kernel_filename,

                const char *kernel_cmdline, const char *initrd_filename,

                const char *cpu_model, enum mainstone_model_e model, int arm_id)

{

    uint32_t sector_len = 256 * 1024;

    target_phys_addr_t mainstone_flash_base[] = { MST_FLASH_0, MST_FLASH_1 };

    struct pxa2xx_state_s *cpu;

    qemu_irq *mst_irq;

    int i, index;



    if (!cpu_model)

        cpu_model = "pxa270-c5";



    /* Setup CPU & memory */

    if (ram_size < MAINSTONE_RAM + MAINSTONE_ROM + 2 * MAINSTONE_FLASH +

                    PXA2XX_INTERNAL_SIZE) {

        fprintf(stderr, "This platform requires %i bytes of memory\n",

                        MAINSTONE_RAM + MAINSTONE_ROM + 2 * MAINSTONE_FLASH +

                        PXA2XX_INTERNAL_SIZE);

        exit(1);

    }



    cpu = pxa270_init(mainstone_binfo.ram_size, cpu_model);

    cpu_register_physical_memory(0, MAINSTONE_ROM,

                    qemu_ram_alloc(MAINSTONE_ROM) | IO_MEM_ROM);



    /* Setup initial (reset) machine state */

    cpu->env->regs[15] = mainstone_binfo.loader_start;



    /* There are two 32MiB flash devices on the board */

    for (i = 0; i < 2; i ++) {

        index = drive_get_index(IF_PFLASH, 0, i);

        if (index == -1) {

            fprintf(stderr, "Two flash images must be given with the "

                    "'pflash' parameter\n");

            exit(1);

        }



        if (!pflash_cfi01_register(mainstone_flash_base[i],

                                qemu_ram_alloc(MAINSTONE_FLASH),

                                drives_table[index].bdrv, sector_len,

                                MAINSTONE_FLASH / sector_len, 4, 0, 0, 0, 0)) {

            fprintf(stderr, "qemu: Error registering flash memory.\n");

            exit(1);

        }

    }



    mst_irq = mst_irq_init(cpu, MST_FPGA_PHYS, PXA2XX_PIC_GPIO_0);



    /* setup keypad */

    printf("map addr %p\n", &map);

    pxa27x_register_keypad(cpu->kp, map, 0xe0);



    /* MMC/SD host */

    pxa2xx_mmci_handlers(cpu->mmc, NULL, mst_irq[MMC_IRQ]);



    smc91c111_init(&nd_table[0], MST_ETH_PHYS, mst_irq[ETHERNET_IRQ]);



    mainstone_binfo.kernel_filename = kernel_filename;

    mainstone_binfo.kernel_cmdline = kernel_cmdline;

    mainstone_binfo.initrd_filename = initrd_filename;

    mainstone_binfo.board_id = arm_id;

    arm_load_kernel(cpu->env, &mainstone_binfo);

}
