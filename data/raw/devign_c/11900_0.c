int rom_add_file(const char *file, const char *fw_dir,

                 target_phys_addr_t addr, int32_t bootindex)

{

    Rom *rom;

    int rc, fd = -1;

    char devpath[100];



    rom = g_malloc0(sizeof(*rom));

    rom->name = g_strdup(file);

    rom->path = qemu_find_file(QEMU_FILE_TYPE_BIOS, rom->name);

    if (rom->path == NULL) {

        rom->path = g_strdup(file);

    }



    fd = open(rom->path, O_RDONLY | O_BINARY);

    if (fd == -1) {

        fprintf(stderr, "Could not open option rom '%s': %s\n",

                rom->path, strerror(errno));

        goto err;

    }



    if (fw_dir) {

        rom->fw_dir  = g_strdup(fw_dir);

        rom->fw_file = g_strdup(file);

    }

    rom->addr    = addr;

    rom->romsize = lseek(fd, 0, SEEK_END);

    rom->data    = g_malloc0(rom->romsize);

    lseek(fd, 0, SEEK_SET);

    rc = read(fd, rom->data, rom->romsize);

    if (rc != rom->romsize) {

        fprintf(stderr, "rom: file %-20s: read error: rc=%d (expected %zd)\n",

                rom->name, rc, rom->romsize);

        goto err;

    }

    close(fd);

    rom_insert(rom);

    if (rom->fw_file && fw_cfg) {

        const char *basename;

        char fw_file_name[56];



        basename = strrchr(rom->fw_file, '/');

        if (basename) {

            basename++;

        } else {

            basename = rom->fw_file;

        }

        snprintf(fw_file_name, sizeof(fw_file_name), "%s/%s", rom->fw_dir,

                 basename);

        fw_cfg_add_file(fw_cfg, fw_file_name, rom->data, rom->romsize);

        snprintf(devpath, sizeof(devpath), "/rom@%s", fw_file_name);

    } else {

        snprintf(devpath, sizeof(devpath), "/rom@" TARGET_FMT_plx, addr);

    }



    add_boot_device_path(bootindex, NULL, devpath);

    return 0;



err:

    if (fd != -1)

        close(fd);

    g_free(rom->data);

    g_free(rom->path);

    g_free(rom->name);

    g_free(rom);

    return -1;

}
