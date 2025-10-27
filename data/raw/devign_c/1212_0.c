build_tpm_tcpa(GArray *table_data, BIOSLinker *linker, GArray *tcpalog)

{

    Acpi20Tcpa *tcpa = acpi_data_push(table_data, sizeof *tcpa);



    tcpa->platform_class = cpu_to_le16(TPM_TCPA_ACPI_CLASS_CLIENT);

    tcpa->log_area_minimum_length = cpu_to_le32(TPM_LOG_AREA_MINIMUM_SIZE);

    acpi_data_push(tcpalog, le32_to_cpu(tcpa->log_area_minimum_length));



    bios_linker_loader_alloc(linker, ACPI_BUILD_TPMLOG_FILE, tcpalog, 1,

                             false /* high memory */);



    /* log area start address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker, ACPI_BUILD_TABLE_FILE,

                                   ACPI_BUILD_TPMLOG_FILE,

                                   &tcpa->log_area_start_address,

                                   sizeof(tcpa->log_area_start_address));



    build_header(linker, table_data,

                 (void *)tcpa, "TCPA", sizeof(*tcpa), 2, NULL, NULL);

}
