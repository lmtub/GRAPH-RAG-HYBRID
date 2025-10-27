void bios_linker_loader_alloc(GArray *linker,

                              const char *file,

                              uint32_t alloc_align,

                              bool alloc_fseg)

{

    BiosLinkerLoaderEntry entry;



    assert(!(alloc_align & (alloc_align - 1)));



    memset(&entry, 0, sizeof entry);

    strncpy(entry.alloc.file, file, sizeof entry.alloc.file - 1);

    entry.command = cpu_to_le32(BIOS_LINKER_LOADER_COMMAND_ALLOCATE);

    entry.alloc.align = cpu_to_le32(alloc_align);

    entry.alloc.zone = cpu_to_le32(alloc_fseg ?

                                    BIOS_LINKER_LOADER_ALLOC_ZONE_FSEG :

                                    BIOS_LINKER_LOADER_ALLOC_ZONE_HIGH);



    /* Alloc entries must come first, so prepend them */

    g_array_prepend_vals(linker, &entry, sizeof entry);

}
