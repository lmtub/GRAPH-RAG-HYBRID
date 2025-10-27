int rom_add_blob(const char *name, const void *blob, size_t len,

                 target_phys_addr_t addr)

{

    Rom *rom;



    rom = g_malloc0(sizeof(*rom));

    rom->name    = g_strdup(name);

    rom->addr    = addr;

    rom->romsize = len;

    rom->data    = g_malloc0(rom->romsize);

    memcpy(rom->data, blob, len);

    rom_insert(rom);

    return 0;

}
