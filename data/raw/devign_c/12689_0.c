void acpi_add_table(GArray *table_offsets, GArray *table_data)

{

    uint32_t offset = cpu_to_le32(table_data->len);

    g_array_append_val(table_offsets, offset);

}
