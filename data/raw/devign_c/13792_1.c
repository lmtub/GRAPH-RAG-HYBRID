static int ds1225y_set_to_mode(ds1225y_t *NVRAM, nvram_open_mode mode, const char *filemode)

{

    if (NVRAM->open_mode != mode)

    {

        if (NVRAM->file)

            qemu_fclose(NVRAM->file);

        NVRAM->file = qemu_fopen(NVRAM->filename, filemode);

        NVRAM->open_mode = mode;

    }

    return (NVRAM->file != NULL);

}
