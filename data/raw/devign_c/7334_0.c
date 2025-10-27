static void verdex_init(MachineState *machine)

{

    const char *cpu_model = machine->cpu_model;

    PXA2xxState *cpu;

    DriveInfo *dinfo;

    int be;

    MemoryRegion *address_space_mem = get_system_memory();



    uint32_t verdex_rom = 0x02000000;

    uint32_t verdex_ram = 0x10000000;



    cpu = pxa270_init(address_space_mem, verdex_ram, cpu_model ?: "pxa270-c0");



    dinfo = drive_get(IF_PFLASH, 0, 0);

    if (!dinfo && !qtest_enabled()) {

        fprintf(stderr, "A flash image must be given with the "

                "'pflash' parameter\n");

        exit(1);

    }



#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    if (!pflash_cfi01_register(0x00000000, NULL, "verdex.rom", verdex_rom,

                               dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL,

                               sector_len, verdex_rom / sector_len,

                               2, 0, 0, 0, 0, be)) {

        fprintf(stderr, "qemu: Error registering flash memory.\n");

        exit(1);

    }



    /* Interrupt line of NIC is connected to GPIO line 99 */

    smc91c111_init(&nd_table[0], 0x04000300,

                    qdev_get_gpio_in(cpu->gpio, 99));

}
