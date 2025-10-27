build_rsdp(GArray *rsdp_table, BIOSLinker *linker, unsigned rsdt)

{

    AcpiRsdpDescriptor *rsdp = acpi_data_push(rsdp_table, sizeof *rsdp);



    bios_linker_loader_alloc(linker, ACPI_BUILD_RSDP_FILE, rsdp_table, 16,

                             true /* fseg memory */);



    memcpy(&rsdp->signature, "RSD PTR ", 8);

    memcpy(rsdp->oem_id, ACPI_BUILD_APPNAME6, 6);

    rsdp->rsdt_physical_address = cpu_to_le32(rsdt);

    /* Address to be filled by Guest linker */

    bios_linker_loader_add_pointer(linker, ACPI_BUILD_RSDP_FILE,

                                   ACPI_BUILD_TABLE_FILE,

                                   &rsdp->rsdt_physical_address,

                                   sizeof rsdp->rsdt_physical_address);

    rsdp->checksum = 0;

    /* Checksum to be filled by Guest linker */

    bios_linker_loader_add_checksum(linker, ACPI_BUILD_RSDP_FILE,

                                    rsdp, sizeof *rsdp,

                                    &rsdp->checksum);



    return rsdp_table;

}
