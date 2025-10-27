void bios_linker_loader_add_pointer(BIOSLinker *linker,

                                    const char *dest_file,

                                    const char *src_file,

                                    void *pointer,

                                    uint8_t pointer_size)

{

    BiosLinkerLoaderEntry entry;

    const BiosLinkerFileEntry *file = bios_linker_find_file(linker, dest_file);

    ptrdiff_t offset = (gchar *)pointer - file->blob->data;



    assert(offset >= 0);

    assert(offset + pointer_size <= file->blob->len);



    memset(&entry, 0, sizeof entry);

    strncpy(entry.pointer.dest_file, dest_file,

            sizeof entry.pointer.dest_file - 1);

    strncpy(entry.pointer.src_file, src_file,

            sizeof entry.pointer.src_file - 1);

    entry.command = cpu_to_le32(BIOS_LINKER_LOADER_COMMAND_ADD_POINTER);

    entry.pointer.offset = cpu_to_le32(offset);

    entry.pointer.size = pointer_size;

    assert(pointer_size == 1 || pointer_size == 2 ||

           pointer_size == 4 || pointer_size == 8);



    g_array_append_vals(linker->cmd_blob, &entry, sizeof entry);

}
