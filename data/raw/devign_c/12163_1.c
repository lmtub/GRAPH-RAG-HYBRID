static inline void acpi_build_tables_cleanup(AcpiBuildTables *tables, bool mfre)

{

    void *linker_data = bios_linker_loader_cleanup(tables->linker);

    if (mfre) {

        g_free(linker_data);

    }

    g_array_free(tables->rsdp, mfre);

    g_array_free(tables->table_data, mfre);

    g_array_free(tables->tcpalog, mfre);

}
