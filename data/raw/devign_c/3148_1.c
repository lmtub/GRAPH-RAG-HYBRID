static void digic_load_rom(DigicBoardState *s, hwaddr addr,

                           hwaddr max_size, const char *def_filename)

{

    target_long rom_size;

    const char *filename;



    if (qtest_enabled()) {

        /* qtest runs no code so don't attempt a ROM load which

         * could fail and result in a spurious test failure.

         */

        return;

    }



    if (bios_name) {

        filename = bios_name;

    } else {

        filename = def_filename;

    }



    if (filename) {

        char *fn = qemu_find_file(QEMU_FILE_TYPE_BIOS, filename);



        if (!fn) {

            error_report("Couldn't find rom image '%s'.", filename);

            exit(1);

        }



        rom_size = load_image_targphys(fn, addr, max_size);

        if (rom_size < 0 || rom_size > max_size) {

            error_report("Couldn't load rom image '%s'.", filename);

            exit(1);

        }


    }

}