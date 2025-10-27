static void nvdimm_build_nfit(GSList *device_list, GArray *table_offsets,

                              GArray *table_data, GArray *linker)

{

    GArray *structures = nvdimm_build_device_structure(device_list);

    void *header;



    acpi_add_table(table_offsets, table_data);



    /* NFIT header. */

    header = acpi_data_push(table_data, sizeof(NvdimmNfitHeader));

    /* NVDIMM device structures. */

    g_array_append_vals(table_data, structures->data, structures->len);



    build_header(linker, table_data, header, "NFIT",

                 sizeof(NvdimmNfitHeader) + structures->len, 1, NULL);

    g_array_free(structures, true);

}
