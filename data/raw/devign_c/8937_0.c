void bios_linker_loader_add_checksum(GArray *linker, const char *file,

                                     GArray *table,

                                     void *start, unsigned size,

                                     uint8_t *checksum)

{

    BiosLinkerLoaderEntry entry;

    ptrdiff_t checksum_offset = (gchar *)checksum - table->data;

    ptrdiff_t start_offset = (gchar *)start - table->data;



    assert(checksum_offset >= 0);

    assert(start_offset >= 0);

    assert(checksum_offset + 1 <= table->len);

    assert(start_offset + size <= table->len);

    assert(*checksum == 0x0);



    memset(&entry, 0, sizeof entry);

    strncpy(entry.cksum.file, file, sizeof entry.cksum.file - 1);

    entry.command = cpu_to_le32(BIOS_LINKER_LOADER_COMMAND_ADD_CHECKSUM);

    entry.cksum.offset = cpu_to_le32(checksum_offset);

    entry.cksum.start = cpu_to_le32(start_offset);

    entry.cksum.length = cpu_to_le32(size);



    g_array_append_vals(linker, &entry, sizeof entry);

}
