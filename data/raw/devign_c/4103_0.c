static unsigned acpi_data_len(GArray *table)

{

#if GLIB_CHECK_VERSION(2, 14, 0)

    assert(g_array_get_element_size(table) == 1);

#endif

    return table->len;

}
