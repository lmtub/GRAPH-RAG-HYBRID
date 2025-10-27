static void acpi_build_update(void *build_opaque, uint32_t offset)

{

    AcpiBuildState *build_state = build_opaque;

    AcpiBuildTables tables;



    /* No state to update or already patched? Nothing to do. */

    if (!build_state || build_state->patched) {

        return;

    }

    build_state->patched = 1;



    acpi_build_tables_init(&tables);



    acpi_build(build_state->guest_info, &tables);



    assert(acpi_data_len(tables.table_data) == build_state->table_size);

    memcpy(build_state->table_ram, tables.table_data->data,

           build_state->table_size);



    acpi_build_tables_cleanup(&tables, true);

}
