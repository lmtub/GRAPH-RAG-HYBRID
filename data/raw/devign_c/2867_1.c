static void test_smbios_ep_table(test_data *data)

{

    struct smbios_entry_point *ep_table = &data->smbios_ep_table;

    uint32_t addr = data->smbios_ep_addr;



    ACPI_READ_ARRAY(ep_table->anchor_string, addr);

    g_assert(!memcmp(ep_table->anchor_string, "_SM_", 4));

    ACPI_READ_FIELD(ep_table->checksum, addr);

    ACPI_READ_FIELD(ep_table->length, addr);

    ACPI_READ_FIELD(ep_table->smbios_major_version, addr);

    ACPI_READ_FIELD(ep_table->smbios_minor_version, addr);

    ACPI_READ_FIELD(ep_table->max_structure_size, addr);

    ACPI_READ_FIELD(ep_table->entry_point_revision, addr);

    ACPI_READ_ARRAY(ep_table->formatted_area, addr);

    ACPI_READ_ARRAY(ep_table->intermediate_anchor_string, addr);

    g_assert(!memcmp(ep_table->intermediate_anchor_string, "_DMI_", 5));

    ACPI_READ_FIELD(ep_table->intermediate_checksum, addr);

    ACPI_READ_FIELD(ep_table->structure_table_length, addr);

    g_assert_cmpuint(ep_table->structure_table_length, >, 0);

    ACPI_READ_FIELD(ep_table->structure_table_address, addr);

    ACPI_READ_FIELD(ep_table->number_of_structures, addr);

    g_assert_cmpuint(ep_table->number_of_structures, >, 0);

    ACPI_READ_FIELD(ep_table->smbios_bcd_revision, addr);

    g_assert(!acpi_checksum((uint8_t *)ep_table, sizeof *ep_table));

    g_assert(!acpi_checksum((uint8_t *)ep_table + 0x10,

                            sizeof *ep_table - 0x10));

}
