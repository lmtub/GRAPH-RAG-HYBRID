static void old_pc_system_rom_init(MemoryRegion *rom_memory, bool isapc_ram_fw)

{

    char *filename;

    MemoryRegion *bios, *isa_bios;

    int bios_size, isa_bios_size;

    int ret;



    /* BIOS load */

    if (bios_name == NULL) {

        bios_name = BIOS_FILENAME;

    }

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    if (filename) {

        bios_size = get_image_size(filename);

    } else {

        bios_size = -1;

    }

    if (bios_size <= 0 ||

        (bios_size % 65536) != 0) {

        goto bios_error;

    }

    bios = g_malloc(sizeof(*bios));

    memory_region_init_ram(bios, NULL, "pc.bios", bios_size, &error_abort);

    vmstate_register_ram_global(bios);

    if (!isapc_ram_fw) {

        memory_region_set_readonly(bios, true);

    }

    ret = rom_add_file_fixed(bios_name, (uint32_t)(-bios_size), -1);

    if (ret != 0) {

    bios_error:

        fprintf(stderr, "qemu: could not load PC BIOS '%s'\n", bios_name);

        exit(1);

    }

    g_free(filename);



    /* map the last 128KB of the BIOS in ISA space */

    isa_bios_size = bios_size;

    if (isa_bios_size > (128 * 1024)) {

        isa_bios_size = 128 * 1024;

    }

    isa_bios = g_malloc(sizeof(*isa_bios));

    memory_region_init_alias(isa_bios, NULL, "isa-bios", bios,

                             bios_size - isa_bios_size, isa_bios_size);

    memory_region_add_subregion_overlap(rom_memory,

                                        0x100000 - isa_bios_size,

                                        isa_bios,

                                        1);

    if (!isapc_ram_fw) {

        memory_region_set_readonly(isa_bios, true);

    }



    /* map all the bios at the top of memory */

    memory_region_add_subregion(rom_memory,

                                (uint32_t)(-bios_size),

                                bios);

}
