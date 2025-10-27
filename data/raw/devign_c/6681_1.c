void acpi_setup(PcGuestInfo *guest_info)

{

    AcpiBuildTables tables;

    AcpiBuildState *build_state;



    if (!guest_info->fw_cfg) {

        ACPI_BUILD_DPRINTF("No fw cfg. Bailing out.\n");

        return;

    }



    if (!guest_info->has_acpi_build) {

        ACPI_BUILD_DPRINTF("ACPI build disabled. Bailing out.\n");

        return;

    }



    if (!acpi_enabled) {

        ACPI_BUILD_DPRINTF("ACPI disabled. Bailing out.\n");

        return;

    }



    build_state = g_malloc0(sizeof *build_state);



    build_state->guest_info = guest_info;



    acpi_set_pci_info();



    acpi_build_tables_init(&tables);

    acpi_build(build_state->guest_info, &tables);



    /* Now expose it all to Guest */

    build_state->table_ram = acpi_add_rom_blob(build_state, tables.table_data,

                                               ACPI_BUILD_TABLE_FILE,

                                               ACPI_BUILD_TABLE_MAX_SIZE);

    assert(build_state->table_ram != RAM_ADDR_MAX);

    build_state->table_size = acpi_data_len(tables.table_data);



    acpi_add_rom_blob(NULL, tables.linker, "etc/table-loader", 0);



    fw_cfg_add_file(guest_info->fw_cfg, ACPI_BUILD_TPMLOG_FILE,

                    tables.tcpalog->data, acpi_data_len(tables.tcpalog));



    /*

     * RSDP is small so it's easy to keep it immutable, no need to

     * bother with ROM blobs.

     */

    fw_cfg_add_file(guest_info->fw_cfg, ACPI_BUILD_RSDP_FILE,

                    tables.rsdp->data, acpi_data_len(tables.rsdp));



    qemu_register_reset(acpi_build_reset, build_state);

    acpi_build_reset(build_state);

    vmstate_register(NULL, 0, &vmstate_acpi_build, build_state);



    /* Cleanup tables but don't free the memory: we track it

     * in build_state.

     */

    acpi_build_tables_cleanup(&tables, false);

}
