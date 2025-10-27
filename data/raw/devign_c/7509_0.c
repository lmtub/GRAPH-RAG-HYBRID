build_facs(GArray *table_data, GArray *linker)

{

    AcpiFacsDescriptorRev1 *facs = acpi_data_push(table_data, sizeof *facs);

    memcpy(&facs->signature, "FACS", 4);

    facs->length = cpu_to_le32(sizeof(*facs));

}
