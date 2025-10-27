int fw_cfg_add_file(FWCfgState *s,  const char *dir, const char *filename,

                    uint8_t *data, uint32_t len)

{

    const char *basename;

    int index;



    if (!s->files) {

        int dsize = sizeof(uint32_t) + sizeof(FWCfgFile) * FW_CFG_FILE_SLOTS;

        s->files = qemu_mallocz(dsize);

        fw_cfg_add_bytes(s, FW_CFG_FILE_DIR, (uint8_t*)s->files, dsize);

    }



    index = be32_to_cpu(s->files->count);

    if (index == FW_CFG_FILE_SLOTS) {

        fprintf(stderr, "fw_cfg: out of file slots\n");

        return 0;

    }



    fw_cfg_add_bytes(s, FW_CFG_FILE_FIRST + index, data, len);



    basename = strrchr(filename, '/');

    if (basename) {

        basename++;

    } else {

        basename = filename;

    }

    if (dir) {

        snprintf(s->files->f[index].name, sizeof(s->files->f[index].name),

                 "%s/%s", dir, basename);

    } else {

        snprintf(s->files->f[index].name, sizeof(s->files->f[index].name),

                 "%s", basename);

    }

    s->files->f[index].size   = cpu_to_be32(len);

    s->files->f[index].select = cpu_to_be16(FW_CFG_FILE_FIRST + index);

    FW_CFG_DPRINTF("%s: #%d: %s (%d bytes)\n", __FUNCTION__,

                   index, s->files->f[index].name, len);



    s->files->count = cpu_to_be32(index+1);

    return 1;

}
