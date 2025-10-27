int rom_add_file(const char *file, const char *fw_dir, const char *fw_file,

                 target_phys_addr_t addr)

{

    Rom *rom;

    int rc, fd = -1;



    rom = qemu_mallocz(sizeof(*rom));

    rom->name = qemu_strdup(file);

    rom->path = qemu_find_file(QEMU_FILE_TYPE_BIOS, rom->name);

    if (rom->path == NULL) {

        rom->path = qemu_strdup(file);

    }



    fd = open(rom->path, O_RDONLY | O_BINARY);

    if (fd == -1) {

        fprintf(stderr, "Could not open option rom '%s': %s\n",

                rom->path, strerror(errno));

        goto err;

    }



    rom->fw_dir  = fw_dir  ? qemu_strdup(fw_dir)  : NULL;

    rom->fw_file = fw_file ? qemu_strdup(fw_file) : NULL;

    rom->addr    = addr;

    rom->romsize = lseek(fd, 0, SEEK_END);

    rom->data    = qemu_mallocz(rom->romsize);

    lseek(fd, 0, SEEK_SET);

    rc = read(fd, rom->data, rom->romsize);

    if (rc != rom->romsize) {

        fprintf(stderr, "rom: file %-20s: read error: rc=%d (expected %zd)\n",

                rom->name, rc, rom->romsize);

        goto err;

    }

    close(fd);

    rom_insert(rom);

    return 0;



err:

    if (fd != -1)

        close(fd);

    qemu_free(rom->data);

    qemu_free(rom->path);

    qemu_free(rom->name);

    qemu_free(rom);

    return -1;

}
