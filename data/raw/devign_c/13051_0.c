static void mainstone_common_init(MemoryRegion *address_space_mem,

                                  MachineState *machine,

                                  enum mainstone_model_e model, int arm_id)

{

    uint32_t sector_len = 256 * 1024;

    hwaddr mainstone_flash_base[] = { MST_FLASH_0, MST_FLASH_1 };

    PXA2xxState *mpu;

    DeviceState *mst_irq;

    DriveInfo *dinfo;

    int i;

    int be;

    MemoryRegion *rom = g_new(MemoryRegion, 1);

    const char *cpu_model = machine->cpu_model;



    if (!cpu_model)

        cpu_model = "pxa270-c5";



    /* Setup CPU & memory */

    mpu = pxa270_init(address_space_mem, mainstone_binfo.ram_size, cpu_model);

    memory_region_init_ram(rom, NULL, "mainstone.rom", MAINSTONE_ROM,

                           &error_abort);

    vmstate_register_ram_global(rom);

    memory_region_set_readonly(rom, true);

    memory_region_add_subregion(address_space_mem, 0, rom);



#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    /* There are two 32MiB flash devices on the board */

    for (i = 0; i < 2; i ++) {

        dinfo = drive_get(IF_PFLASH, 0, i);

        if (!dinfo) {

            if (qtest_enabled()) {

                break;

            }

            fprintf(stderr, "Two flash images must be given with the "

                    "'pflash' parameter\n");

            exit(1);

        }



        if (!pflash_cfi01_register(mainstone_flash_base[i], NULL,

                                   i ? "mainstone.flash1" : "mainstone.flash0",

                                   MAINSTONE_FLASH,

                                   blk_bs(blk_by_legacy_dinfo(dinfo)),

                                   sector_len, MAINSTONE_FLASH / sector_len,

                                   4, 0, 0, 0, 0, be)) {

            fprintf(stderr, "qemu: Error registering flash memory.\n");

            exit(1);

        }

    }



    mst_irq = sysbus_create_simple("mainstone-fpga", MST_FPGA_PHYS,

                    qdev_get_gpio_in(mpu->gpio, 0));



    /* setup keypad */

    pxa27x_register_keypad(mpu->kp, map, 0xe0);



    /* MMC/SD host */

    pxa2xx_mmci_handlers(mpu->mmc, NULL, qdev_get_gpio_in(mst_irq, MMC_IRQ));



    pxa2xx_pcmcia_set_irq_cb(mpu->pcmcia[0],

            qdev_get_gpio_in(mst_irq, S0_IRQ),

            qdev_get_gpio_in(mst_irq, S0_CD_IRQ));

    pxa2xx_pcmcia_set_irq_cb(mpu->pcmcia[1],

            qdev_get_gpio_in(mst_irq, S1_IRQ),

            qdev_get_gpio_in(mst_irq, S1_CD_IRQ));



    smc91c111_init(&nd_table[0], MST_ETH_PHYS,

                    qdev_get_gpio_in(mst_irq, ETHERNET_IRQ));



    mainstone_binfo.kernel_filename = machine->kernel_filename;

    mainstone_binfo.kernel_cmdline = machine->kernel_cmdline;

    mainstone_binfo.initrd_filename = machine->initrd_filename;

    mainstone_binfo.board_id = arm_id;

    arm_load_kernel(mpu->cpu, &mainstone_binfo);

}
