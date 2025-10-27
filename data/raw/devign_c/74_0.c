static void virt_acpi_build_update(void *build_opaque)

{

    AcpiBuildState *build_state = build_opaque;

    AcpiBuildTables tables;



    /* No state to update or already patched? Nothing to do. */

    if (!build_state || build_state->patched) {

        return;

    }

    build_state->patched = true;



    acpi_build_tables_init(&tables);



    virt_acpi_build(build_state->guest_info, &tables);



    acpi_ram_update(build_state->table_mr, tables.table_data);

    acpi_ram_update(build_state->rsdp_mr, tables.rsdp);

    acpi_ram_update(build_state->linker_mr, tables.linker);





    acpi_build_tables_cleanup(&tables, true);

}
