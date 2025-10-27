static void acpi_build_update(void *build_opaque)

{

    AcpiBuildState *build_state = build_opaque;

    AcpiBuildTables tables;



    /* No state to update or already patched? Nothing to do. */

    if (!build_state || build_state->patched) {

        return;

    }

    build_state->patched = 1;



    acpi_build_tables_init(&tables);



    acpi_build(&tables, MACHINE(qdev_get_machine()));



    acpi_ram_update(build_state->table_mr, tables.table_data);



    if (build_state->rsdp) {

        memcpy(build_state->rsdp, tables.rsdp->data, acpi_data_len(tables.rsdp));

    } else {

        acpi_ram_update(build_state->rsdp_mr, tables.rsdp);

    }



    acpi_ram_update(build_state->linker_mr, tables.linker);

    acpi_build_tables_cleanup(&tables, true);

}
