void bios_linker_loader_add_checksum(GArray *linker, const char *file,

                                     void *table,

                                     void *start, unsigned size,

                                     uint8_t *checksum)

{

    BiosLinkerLoaderEntry entry;



    memset(&entry, 0, sizeof entry);

    strncpy(entry.cksum.file, file, sizeof entry.cksum.file - 1);

    entry.command = cpu_to_le32(BIOS_LINKER_LOADER_COMMAND_ADD_CHECKSUM);

    entry.cksum.offset = cpu_to_le32(checksum - (uint8_t *)table);

    entry.cksum.start = cpu_to_le32((uint8_t *)start - (uint8_t *)table);

    entry.cksum.length = cpu_to_le32(size);



    g_array_append_val(linker, entry);

}
