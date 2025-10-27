build_rsdt(GArray *table_data, GArray *linker, GArray *table_offsets,

           const char *oem_id, const char *oem_table_id)

{

    AcpiRsdtDescriptorRev1 *rsdt;

    size_t rsdt_len;

    int i;

    const int table_data_len = (sizeof(uint32_t) * table_offsets->len);



    rsdt_len = sizeof(*rsdt) + table_data_len;

    rsdt = acpi_data_push(table_data, rsdt_len);

    memcpy(rsdt->table_offset_entry, table_offsets->data, table_data_len);

    for (i = 0; i < table_offsets->len; ++i) {

        /* rsdt->table_offset_entry to be filled by Guest linker */

        bios_linker_loader_add_pointer(linker,

                                       ACPI_BUILD_TABLE_FILE,

                                       ACPI_BUILD_TABLE_FILE,

                                       table_data, &rsdt->table_offset_entry[i],

                                       sizeof(uint32_t));

    }

    build_header(linker, table_data,

                 (void *)rsdt, "RSDT", rsdt_len, 1, oem_id, oem_table_id);

}
