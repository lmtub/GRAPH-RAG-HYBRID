int rom_add_file(const char *file, const char *fw_dir,
                 hwaddr addr, int32_t bootindex,
                 bool option_rom, MemoryRegion *mr)
{
    MachineClass *mc = MACHINE_GET_CLASS(qdev_get_machine());
    Rom *rom;
    int rc, fd = -1;
    char devpath[100];
    rom = g_malloc0(sizeof(*rom));
    rom->name = g_strdup(file);
    rom->path = qemu_find_file(QEMU_FILE_TYPE_BIOS, rom->name);
    if (rom->path == NULL) {
        rom->path = g_strdup(file);
    fd = open(rom->path, O_RDONLY | O_BINARY);
    if (fd == -1) {
        fprintf(stderr, "Could not open option rom '%s': %s\n",
                rom->path, strerror(errno));
        goto err;
        rom->fw_dir  = g_strdup(fw_dir);
        rom->fw_file = g_strdup(file);
    rom->addr     = addr;
    rom->romsize  = lseek(fd, 0, SEEK_END);
    if (rom->romsize == -1) {
        fprintf(stderr, "rom: file %-20s: get size error: %s\n",
                rom->name, strerror(errno));
        goto err;
    rom->datasize = rom->romsize;
    rom->data     = g_malloc0(rom->datasize);
    lseek(fd, 0, SEEK_SET);
    rc = read(fd, rom->data, rom->datasize);
    if (rc != rom->datasize) {
        fprintf(stderr, "rom: file %-20s: read error: rc=%d (expected %zd)\n",
                rom->name, rc, rom->datasize);
        goto err;
    close(fd);
    rom_insert(rom);
    if (rom->fw_file && fw_cfg) {
        const char *basename;
        char fw_file_name[FW_CFG_MAX_FILE_PATH];
        void *data;
        basename = strrchr(rom->fw_file, '/');
        if (basename) {
            basename++;
        } else {
            basename = rom->fw_file;
        snprintf(fw_file_name, sizeof(fw_file_name), "%s/%s", rom->fw_dir,
                 basename);
        snprintf(devpath, sizeof(devpath), "/rom@%s", fw_file_name);
        if ((!option_rom || mc->option_rom_has_mr) && mc->rom_file_has_mr) {
            data = rom_set_mr(rom, OBJECT(fw_cfg), devpath);
        } else {
            data = rom->data;
        fw_cfg_add_file(fw_cfg, fw_file_name, data, rom->romsize);
    } else {
        if (mr) {
            rom->mr = mr;
            snprintf(devpath, sizeof(devpath), "/rom@%s", file);
        } else {
            snprintf(devpath, sizeof(devpath), "/rom@" TARGET_FMT_plx, addr);
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